//! PostgreSQL store for Clypeus.
//!
//! Owns the schema migrations and connects the shared SQL implementation to a
//! PostgreSQL database. Use `postgresql://` URLs.

use clypeus_core::store::StoreError;
use clypeus_store_sql::{SqlDialect, SqlStore};

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Connects to PostgreSQL and applies the Clypeus schema.
pub async fn connect(url: &str) -> Result<SqlStore, StoreError> {
    SqlStore::connect(url, SqlDialect::Postgres, &MIGRATOR).await
}
