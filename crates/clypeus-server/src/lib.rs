//! Library surface of the standalone Clypeus server.
//!
//! The `clypeus` binary is a thin wrapper around [`app::router`]; integration
//! tests and embedders can build the same router with a custom configuration.

pub mod app;
pub mod config;
pub mod demo;
pub mod dto;
pub mod error;
pub mod handlers;
pub mod openapi;
pub mod state;
