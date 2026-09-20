//! Clypeus standalone server (`clypeus`).
//!
//! Starts an HTTP server exposing the core API over a configured store,
//! principal resolver, and provider set. `clypeus --print-openapi` prints the
//! OpenAPI document and exits.

use std::sync::Arc;

use clypeus_server::config::ServerConfig;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--print-openapi") {
        print!(
            "{}",
            serde_json::to_string_pretty(&clypeus_server::openapi::spec()).unwrap_or_default()
        );
        return;
    }
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "clypeus {}\n\nUSAGE:\n    clypeus [--print-openapi]\n\nENVIRONMENT:\n    \
             CLYPEUS_BIND_ADDR, CLYPEUS_STORE, CLYPEUS_DATABASE_URL, CLYPEUS_SQLITE_PATH,\n    \
             CLYPEUS_STATIC_TOKEN, CLYPEUS_STATIC_SCOPE, CLYPEUS_JWKS_PATH, CLYPEUS_SECRET_DIR,\n    \
             CLYPEUS_ADMIN_SCOPES, CLYPEUS_ALLOW_PRIVATE_PROVIDERS, CLYPEUS_* (limits)\n",
            env!("CARGO_PKG_VERSION")
        );
        return;
    }
    if let Err(error) = run() {
        eprintln!("clypeus: {error}");
        std::process::exit(1);
    }
}

#[tokio::main]
async fn run() -> Result<(), String> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .map_err(|error| format!("metrics recorder: {error}"))?;
    clypeus_core::metrics::init();

    let config = ServerConfig::from_env()?;
    let bind_addr = config.bind_addr;
    let state = clypeus_server::state::AppState::build(config, metrics).await?;
    let app = clypeus_server::app::router(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .map_err(|error| format!("bind {bind_addr}: {error}"))?;
    tracing::info!(%bind_addr, "clypeus listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| format!("server: {error}"))?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutting down");
}
