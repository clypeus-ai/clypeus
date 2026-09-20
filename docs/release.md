# Release process

Releases are immutable: a version tag never moves after publication. The tag
points at the exact commit whose `Cargo.lock` is part of the release, and
consumers pin that tag in their manifests.

## Versioning

* SemVer across the crate set; every crate carries the workspace version.
* A breaking core change bumps the major version and moves the HTTP surface to
  a new prefix (`/v1` → `/v2`). The previous major receives fixes for one
  minor release.
* Pre-1.0 history is tag-only (`v0.1.0` … `v0.1.3`); `v1.0.0` is the first
  stable release.

## Cutting a release

1. Confirm the gates are green:

   ```bash
   ./scripts/ci.sh
   ```

2. Update `CHANGELOG.md` and the workspace version in `Cargo.toml`, then
   refresh the lockfile:

   ```bash
   cargo update --workspace
   cargo build --workspace --locked
   ```

3. Commit with a conventional message and tag the commit:

   ```bash
   git tag -a v1.0.0 -m "Clypeus v1.0.0"
   git push origin main v1.0.0
   ```

4. Build, sign, and publish the image with `scripts/release.sh` (below).

## Signed image, SBOM, and referrers

`scripts/release.sh` builds `release/Dockerfile`, generates a CycloneDX SBOM
with syft, attaches the SBOM to the image, signs the image with cosign, and
verifies the signature before it exits successfully. The pinned tool versions
live in `release/tools.lock.toml`.

The signing key **must live outside the repository**. Generate it once and
keep it in a secret manager or an encrypted store:

```bash
cosign generate-key-pair --output-key-prefix ~/.config/clypeus/cosign
# writes ~/.config/clypeus/cosign.key (private) and cosign.pub (public)
```

Publish the public half with the release notes so consumers can verify:

```bash
export CLYPEUS_RELEASE_REGISTRY=ghcr.io/<owner>/clypeus
export COSIGN_KEY=~/.config/clypeus/cosign.key
export COSIGN_PASSWORD=...          # read from your secret store

scripts/release.sh 1.0.0

# consumers verify against the published public key
cosign verify --key cosign.pub ghcr.io/<owner>/clypeus:v1.0.0
cosign verify-attestation ...       # SBOM referrer
```

The script works against any OCI registry that supports the distribution
spec and, for signatures and SBOM attach, the referrers API. Private
registries with a private CA are supported through
`CLYPEUS_RELEASE_CA=/path/to/ca.crt`.

## GitHub Actions

`.github/workflows/*` can only be pushed by a token that carries the GitHub
`workflow` scope. The canonical workflows live in `ci/`:

* `ci/github-ci.yml` — every push and pull request.
* `ci/github-release.yml` — tag pushes: build, Syft SBOM, cosign keyless
  signing through the GitHub OIDC identity, push to GHCR, release notes.

Install them (or let the release workflow be installed once) with:

```bash
gh auth refresh -h github.com -s workflow
mkdir -p .github/workflows
cp ci/github-ci.yml .github/workflows/ci.yml
cp ci/github-release.yml .github/workflows/release.yml
git add .github/workflows && git commit -m "ci: install the canonical workflows"
git push
```

The local `scripts/release.sh` is the same pipeline with a local signing key;
the GitHub workflow uses keyless OIDC instead and needs no secret.

## Consumer pinning

Consumers depend on the repository by tag and commit the resolved revision in
`Cargo.lock`:

```toml
clypeus-core = { git = "https://github.com/vitkuz573/clypeus", tag = "v1.0.0" }
```

A consumer's release gate should fail when the tag resolves to a different
revision than the one recorded in the lockfile, which `cargo build --locked`
and `cargo metadata --locked` enforce.
