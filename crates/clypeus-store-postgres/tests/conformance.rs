//! Runs the shared conformance suite against PostgreSQL.
//!
//! Requires `CLYPEUS_TEST_DATABASE_URL`; the test is skipped when unset so a
//! checkout without a database stays green.

use std::sync::Arc;

use clypeus_conformance::{run_broker_conformance, run_store_conformance};
use clypeus_core::audit::AuditSink;

#[tokio::test]
async fn postgres_store_passes_conformance() {
    let Ok(url) = std::env::var("CLYPEUS_TEST_DATABASE_URL") else {
        eprintln!("CLYPEUS_TEST_DATABASE_URL is not set; skipping");
        return;
    };
    let store = Arc::new(
        clypeus_store_postgres::connect(&url)
            .await
            .expect("postgres store connects"),
    );
    run_store_conformance(store.as_ref())
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let audit: Arc<dyn AuditSink> = store.clone();
    run_broker_conformance(store, audit)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}
