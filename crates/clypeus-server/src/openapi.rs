//! OpenAPI document generated from the handler annotations.

use utoipa::OpenApi;

/// Clypeus standalone API.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Clypeus API",
        version = "0.1.0",
        description = "Policy-first AI gateway: provider completions, conversations, \
                       typed tool egress, approvals, and an append-only audit trail.",
        license(name = "MIT OR Apache-2.0")
    ),
    paths(
        crate::handlers::healthz,
        crate::handlers::readyz,
        crate::handlers::metrics,
        crate::handlers::list_models,
        crate::handlers::preview_models,
        crate::handlers::completions,
        crate::handlers::list_threads,
        crate::handlers::create_thread,
        crate::handlers::get_thread,
        crate::handlers::update_thread,
        crate::handlers::delete_thread,
        crate::handlers::thread_usage,
        crate::handlers::create_message,
        crate::handlers::edit_message,
        crate::handlers::regenerate_message,
        crate::handlers::activate_message,
        crate::handlers::set_feedback,
        crate::handlers::clear_feedback,
        crate::handlers::submit_approval,
        crate::handlers::list_tools,
        crate::handlers::list_functions,
        crate::handlers::run_function,
        crate::handlers::list_audit,
        crate::handlers::export_audit,
        crate::handlers::get_scope_settings,
        crate::handlers::put_scope_settings,
        crate::handlers::test_scope_settings,
        crate::handlers::admin_preview_models,
    ),
    components(schemas(
        crate::error::Problem,
        crate::dto::ThreadDto,
        crate::dto::ThreadListResponse,
        crate::dto::ThreadViewResponse,
        crate::dto::MessageDto,
        crate::dto::MessageVersionDto,
        crate::dto::ToolCallDto,
        crate::dto::ToolCallErrorDto,
        crate::dto::ApprovalDto,
        crate::dto::FeedbackDto,
        crate::dto::TurnResponse,
        crate::dto::CreateThreadRequest,
        crate::dto::UpdateThreadRequest,
        crate::dto::CreateTurnRequest,
        crate::dto::EditTurnRequest,
        crate::dto::RegenerateTurnRequest,
        crate::dto::SubmitApprovalRequest,
        crate::dto::SetFeedbackRequest,
        crate::dto::PageContextDto,
        crate::dto::ScopeSettingsDto,
        crate::dto::UpdateScopeSettingsRequest,
        crate::dto::ConnectionTestDto,
        crate::dto::ModelsPreviewRequest,
        crate::dto::ModelsResponse,
        crate::dto::CompletionsRequest,
        crate::dto::AuditListResponse,
        crate::dto::UsageResponse,
        crate::dto::UsageEntryDto,
        crate::dto::ToolCatalogEntryDto,
        crate::dto::ToolCatalogResponse,
        clypeus_core::functions::FunctionDescriptorDto,
        clypeus_core::functions::FunctionListResponse,
        clypeus_core::functions::RunFunctionRequest,
        clypeus_core::functions::RunFunctionResponse,
        clypeus_core::functions::FunctionDiagnosticsDto,
        clypeus_core::audit::AuditRecord,
        clypeus_core::audit::AuditItemKind,
        clypeus_core::models::ChatMessage,
        clypeus_core::models::ChatRole,
        clypeus_core::models::ProviderKind,
        clypeus_core::models::TokenUsage,
        clypeus_core::models::ToolSpec,
        clypeus_core::models::ToolCall,
        clypeus_core::provider::ModelCapability,
        clypeus_core::profile::ProfileSelection,
    ))
)]
#[derive(Debug)]
pub struct ApiDoc;

/// Serializes the document.
pub fn spec() -> serde_json::Value {
    serde_json::to_value(ApiDoc::openapi()).unwrap_or(serde_json::Value::Null)
}
