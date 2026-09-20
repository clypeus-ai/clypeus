//! Runs the shared conformance suite against a SQLite database.

use std::sync::Arc;

use clypeus_conformance::{run_broker_conformance, run_store_conformance};
use clypeus_core::audit::AuditSink;

async fn store() -> Arc<clypeus_store_sql::SqlStore> {
    let path = std::env::temp_dir().join(format!("clypeus-{}.db", uuid::Uuid::new_v4()));
    let url = format!("sqlite://{}?mode=rwc", path.to_string_lossy());
    let store = clypeus_store_sqlite::connect(&url)
        .await
        .expect("sqlite store connects");
    Arc::new(store)
}

#[tokio::test]
async fn sqlite_store_passes_conformance() {
    let store = store().await;
    run_store_conformance(store.as_ref())
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
async fn sqlite_store_passes_broker_conformance() {
    let store = store().await;
    let audit: Arc<dyn AuditSink> = store.clone();
    run_broker_conformance(store, audit)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}
