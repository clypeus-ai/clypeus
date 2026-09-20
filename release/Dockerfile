# Build the standalone Clypeus server and ship it on a slim runtime image.
FROM rust:1-slim-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p clypeus-server --bin clypeus

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /var/lib/clypeus clypeus \
    && mkdir -p /var/lib/clypeus/secrets \
    && chown -R clypeus:clypeus /var/lib/clypeus
COPY --from=builder /src/target/release/clypeus /usr/local/bin/clypeus
USER clypeus
ENV CLYPEUS_BIND_ADDR=0.0.0.0:8080 \
    CLYPEUS_STORE=postgres \
    CLYPEUS_SECRET_DIR=/var/lib/clypeus/secrets \
    RUST_LOG=info
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/clypeus"]
