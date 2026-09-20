//! Runs the shared conformance suite against the in-memory store.

use std::sync::Arc;

use clypeus_conformance::{
    run_broker_conformance, run_egress_conformance, run_guard_conformance, run_store_conformance,
};
use clypeus_core::audit::AuditSink;
use clypeus_store_memory::MemoryStore;

#[tokio::test]
async fn memory_store_passes_conformance() {
    let store = MemoryStore::new();
    run_store_conformance(&store)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    run_egress_conformance().unwrap_or_else(|error| panic!("{error}"));
    run_guard_conformance().unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
async fn memory_store_passes_broker_conformance() {
    let store = Arc::new(MemoryStore::new());
    let audit: Arc<dyn AuditSink> = store.clone();
    run_broker_conformance(store, audit)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}
