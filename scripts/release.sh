#!/usr/bin/env bash
# Build, sign, and publish the release image.
#
# Usage: scripts/release.sh <version>
#
# Environment:
#   CLYPEUS_RELEASE_REGISTRY  required; host/repository, e.g. ghcr.io/acme/clypeus
#   COSIGN_KEY                required; path to the cosign private key (outside the repo)
#   COSIGN_PASSWORD           required by cosign when the key is encrypted
#   CLYPEUS_RELEASE_CA        optional; CA bundle for a registry with a private CA
#   CLYPEUS_RELEASE_BUILDER   optional; buildx builder name
#   CLYPEUS_RELEASE_SKIP_TLOG optional; "false" uploads to the Rekor transparency log
#
# The pipeline: buildx build --push, CycloneDX SBOM with syft, cosign attach
# sbom, cosign sign, cosign verify. Tool versions are pinned in
# release/tools.lock.toml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="${1:?usage: scripts/release.sh <version>}"
registry="${CLYPEUS_RELEASE_REGISTRY:?set CLYPEUS_RELEASE_REGISTRY to host/repository}"
key="${COSIGN_KEY:?set COSIGN_KEY to the cosign private key path}"
pub="${COSIGN_PUBLIC_KEY:-${key%.key}.pub}"
[[ -f "$key" ]] || { echo "cosign key not found: $key" >&2; exit 1; }
[[ -f "$pub" ]] || { echo "cosign public key not found: $pub" >&2; exit 1; }

tools="$ROOT/release/tools.lock.toml"
tool_image() {
    python3 - "$tools" "$1" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
match = re.search(rf'^{re.escape(sys.argv[2])}\s*=\s*"([^"]+)"', text, re.M)
if not match:
    raise SystemExit(f"pinned image for {sys.argv[2]} not found")
print(match.group(1))
PY
}
syft_image="$(tool_image syft)"
cosign_image="$(tool_image cosign)"

manifest_version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
[[ "$version" == "$manifest_version" ]] || {
    echo "version $version does not match Cargo.toml ($manifest_version)" >&2
    exit 1
}

git -C "$ROOT" diff --quiet || { echo "working tree is dirty" >&2; exit 1; }
if git -C "$ROOT" rev-parse -q --verify "refs/tags/v$version" >/dev/null; then
    echo "tag v$version already exists; releases are immutable" >&2
    exit 1
fi

mounts=()
run_mounts=()
if [[ -n "${CLYPEUS_RELEASE_CA:-}" ]]; then
    mounts+=(-v "${CLYPEUS_RELEASE_CA}:/clypeus-registry-ca.crt:ro")
    run_mounts+=(-e SSL_CERT_FILE=/clypeus-registry-ca.crt)
fi
docker_config="${DOCKER_CONFIG:-$HOME/.docker}"
if [[ -d "$docker_config" ]]; then
    run_mounts+=(-v "$docker_config:/docker-config:ro" -e DOCKER_CONFIG=/docker-config)
fi

builder_args=()
if [[ -n "${CLYPEUS_RELEASE_BUILDER:-}" ]]; then
    builder_args+=(--builder "$CLYPEUS_RELEASE_BUILDER")
fi

echo "==> build and push $registry:$version"
docker buildx build "${builder_args[@]}" \
    --file "$ROOT/release/Dockerfile" \
    --platform linux/amd64 \
    --provenance=false \
    --push \
    --tag "$registry:$version" \
    --tag "$registry:latest" \
    --metadata-file /tmp/clypeus-release-metadata.json \
    "$ROOT"
digest="$(python3 -c 'import json; print(json.load(open("/tmp/clypeus-release-metadata.json"))["containerimage.digest"])')"
image="$registry@$digest"
echo "    digest $digest"

sbom="$(mktemp -t clypeus-sbom-XXXXXX.cdx.json)"
echo "==> SBOM (syft $syft_image)"
docker run --rm "${mounts[@]}" "${run_mounts[@]}" \
    -v "$sbom:/sbom.cdx.json" \
    "$syft_image" "registry:$registry@$digest" -o cyclonedx-json=/sbom.cdx.json
[[ -s "$sbom" ]] || { echo "syft produced no SBOM" >&2; exit 1; }

echo "==> attach SBOM"
docker run --rm "${mounts[@]}" "${run_mounts[@]}" \
    -v "$sbom:/sbom.cdx.json:ro" \
    -v "$key:/cosign.key:ro" \
    -e COSIGN_PASSWORD="${COSIGN_PASSWORD:-}" \
    "$cosign_image" attach sbom --sbom /sbom.cdx.json --type cyclonedx "$image"

echo "==> sign"
tlog_args=(--tlog-upload=false)
verify_args=(--insecure-ignore-tlog=true)
if [[ "${CLYPEUS_RELEASE_SKIP_TLOG:-true}" == "false" ]]; then
    tlog_args=()
    verify_args=()
fi
docker run --rm "${mounts[@]}" "${run_mounts[@]}" \
    -v "$key:/cosign.key:ro" \
    -e COSIGN_PASSWORD="${COSIGN_PASSWORD:-}" \
    "$cosign_image" sign --key /cosign.key --yes "${tlog_args[@]}" "$image"

echo "==> verify"
docker run --rm "${mounts[@]}" "${run_mounts[@]}" \
    -v "$pub:/cosign.pub:ro" \
    "$cosign_image" verify --key /cosign.pub "${verify_args[@]}" "$image" >/dev/null

echo "release $registry:$version"
echo "digest  $digest"
echo "sbom    $sbom"
