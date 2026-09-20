//! Router assembly.

use std::sync::Arc;

use axum::Router;
use axum::middleware;
use axum::routing::{get, patch, post, put};

use crate::handlers;
use crate::state::AppState;

/// Builds the public and authenticated routers.
pub fn router(state: Arc<AppState>) -> Router {
    let public = Router::new()
        .route("/healthz", get(handlers::healthz))
        .route("/readyz", get(handlers::readyz))
        .route("/metrics", get(handlers::metrics))
        .route("/openapi/v1.json", get(openapi_json));

    let api = Router::new()
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/preview", post(handlers::preview_models))
        .route("/v1/completions", post(handlers::completions))
        .route(
            "/v1/threads",
            get(handlers::list_threads).post(handlers::create_thread),
        )
        .route(
            "/v1/threads/{thread_id}",
            get(handlers::get_thread)
                .patch(handlers::update_thread)
                .delete(handlers::delete_thread),
        )
        .route(
            "/v1/threads/{thread_id}/messages",
            post(handlers::create_message),
        )
        .route("/v1/threads/{thread_id}/usage", get(handlers::thread_usage))
        .route("/v1/messages/{message_id}", patch(handlers::edit_message))
        .route(
            "/v1/messages/{message_id}/regenerate",
            post(handlers::regenerate_message),
        )
        .route(
            "/v1/messages/{message_id}/activate",
            post(handlers::activate_message),
        )
        .route(
            "/v1/messages/{message_id}/feedback",
            put(handlers::set_feedback).delete(handlers::clear_feedback),
        )
        .route(
            "/v1/tool-calls/{tool_call_id}/approvals",
            post(handlers::submit_approval),
        )
        .route("/v1/tools", get(handlers::list_tools))
        .route("/v1/functions", get(handlers::list_functions))
        .route("/v1/functions/{name}", post(handlers::run_function))
        .route("/v1/audit", get(handlers::list_audit))
        .route("/v1/audit/export", get(handlers::export_audit))
        .route(
            "/admin/v1/scopes/{scope}/settings",
            get(handlers::get_scope_settings).put(handlers::put_scope_settings),
        )
        .route(
            "/admin/v1/scopes/{scope}/settings/test",
            post(handlers::test_scope_settings),
        )
        .route(
            "/admin/v1/scopes/{scope}/models/preview",
            post(handlers::admin_preview_models),
        )
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            handlers::auth_middleware,
        ));

    Router::new().merge(public).merge(api).with_state(state)
}

async fn openapi_json() -> axum::Json<serde_json::Value> {
    axum::Json(crate::openapi::spec())
}
