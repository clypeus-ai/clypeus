//! HTTP surface of the standalone server.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use clypeus_core::audit::AuditQuery;
use clypeus_core::broker::{ApprovalGrant, Caller, ToolCallRequest};
use clypeus_core::context::TurnContext;
use clypeus_core::functions::RunFunctionRequest;
use clypeus_core::models::{ChatMessage, ChatRole, ProviderKind, ReasoningLevel, ToolSpec};
use clypeus_core::orchestrator::{
    ResumeAction, ResumeProvider, ResumeRequest, TurnRequest, budget_for,
};
use clypeus_core::principal::{AuthRequest, Principal, ScopeId};
use clypeus_core::profile::ProfileSelection;
use clypeus_core::provider::{ProviderConfig, validate_base_url};
use clypeus_core::rate_limit::RateKey;
use clypeus_core::store::{
    CreateThread, MessageStatus, ScopeSettings, ScopeSettingsUpdate, ThreadUpdate, ToolCallStatus,
    TurnTarget as StoreTurnTarget,
};
use clypeus_core::tools::ToolError;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::dto::*;
use crate::error::{ApiError, Problem, probe_body};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Authentication and authorization
// ---------------------------------------------------------------------------

/// Resolves the principal and stores it in request extensions.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    mut request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let hint = path_scope_hint(request.uri().path());
    let headers = request.headers().clone();
    let principal = {
        let auth = AuthRequest {
            headers: &headers,
            path_scope_hint: hint.as_deref(),
        };
        state.resolver.resolve(&auth).await?
    };
    request.extensions_mut().insert(principal);
    Ok(next.run(request).await)
}

fn path_scope_hint(path: &str) -> Option<String> {
    let mut segments = path.split('/').filter(|segment| !segment.is_empty());
    while let Some(segment) = segments.next() {
        if segment == "scopes" {
            return segments.next().map(str::to_string);
        }
    }
    None
}

fn require_admin(state: &AppState, principal: &Principal) -> Result<(), ApiError> {
    if state.policy.is_admin(principal) {
        Ok(())
    } else {
        Err(ApiError::forbidden(
            "admin_required",
            "Administrative scope is required.",
        ))
    }
}

fn check_rate_limit(state: &AppState, principal: &Principal) -> Result<(), ApiError> {
    state
        .chat_limiter
        .check(&RateKey::new(
            principal.scope.clone(),
            principal.subject.clone(),
            "chat",
        ))
        .map(|_| ())
        .map_err(|error| {
            ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                error.to_string(),
            )
        })
}

// ---------------------------------------------------------------------------
// Probes and metrics
// ---------------------------------------------------------------------------

#[utoipa::path(get, path = "/healthz", tag = "Operations", responses((status = 200, description = "Liveness")))]
pub async fn healthz() -> Json<Value> {
    probe_body("ok", None)
}

#[utoipa::path(get, path = "/readyz", tag = "Operations", responses((status = 200, description = "Readiness"), (status = 503, description = "Unavailable")))]
pub async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    match state.readiness.check_ready().await {
        Ok(()) => (StatusCode::OK, probe_body("ready", None)).into_response(),
        Err(detail) => (
            StatusCode::SERVICE_UNAVAILABLE,
            probe_body("unavailable", Some(&detail)),
        )
            .into_response(),
    }
}

#[utoipa::path(get, path = "/metrics", tag = "Operations", responses((status = 200, description = "Prometheus metrics")))]
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Provider catalog
// ---------------------------------------------------------------------------

async fn resolve_provider(
    state: &AppState,
    scope: &ScopeId,
) -> Result<(ScopeSettings, ProviderConfig), ApiError> {
    let settings = state.settings.get(scope).await?.ok_or_else(|| {
        ApiError::unavailable(
            "provider_not_configured",
            "AI provider is not configured for this scope.",
        )
    })?;
    let Some(base_url) = settings.base_url.clone() else {
        return Err(ApiError::unavailable(
            "provider_not_configured",
            "AI provider base URL is not configured.",
        ));
    };
    let api_key = state
        .secrets
        .get(scope, clypeus_core::functions::PROVIDER_API_KEY)
        .await
        .map_err(|_| ApiError::internal("Provider credentials are unavailable."))?
        .ok_or_else(|| {
            ApiError::unavailable(
                "provider_not_configured",
                "AI provider API key is not configured.",
            )
        })?;
    let config = ProviderConfig {
        base_url,
        api_key,
        timeout: Duration::from_millis(u64::try_from(settings.timeout_ms).unwrap_or(60_000)),
        max_output_tokens: settings.max_output_tokens,
        allow_private_targets: state.config.core.allow_private_providers,
    };
    Ok((settings, config))
}

async fn fetch_catalog(
    state: &AppState,
    provider_kind: ProviderKind,
    config: &ProviderConfig,
) -> Result<clypeus_core::provider::ModelCatalog, ApiError> {
    let provider = state.providers.get(provider_kind).ok_or_else(|| {
        ApiError::unavailable(
            "provider_not_configured",
            "Provider backend is not configured.",
        )
    })?;
    provider.catalog(config).await.map_err(Into::into)
}

#[utoipa::path(get, path = "/v1/models", tag = "Models", responses((status = 200, description = "Model catalog", body = ModelsResponse)))]
pub async fn list_models(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<ModelsResponse>, ApiError> {
    let (settings, config) = resolve_provider(&state, &principal.scope).await?;
    let catalog = fetch_catalog(&state, settings.provider_kind, &config).await?;
    Ok(Json(catalog.into()))
}

#[utoipa::path(post, path = "/v1/models/preview", tag = "Models", request_body = ModelsPreviewRequest, responses((status = 200, description = "Model catalog", body = ModelsResponse)))]
pub async fn preview_models(
    State(state): State<Arc<AppState>>,
    Extension(_principal): Extension<Principal>,
    Json(request): Json<ModelsPreviewRequest>,
) -> Result<Json<ModelsResponse>, ApiError> {
    validate_base_url(&request.base_url, state.config.core.allow_private_providers)?;
    let config = ProviderConfig {
        base_url: request.base_url,
        api_key: request.api_key.into(),
        timeout: Duration::from_secs(30),
        max_output_tokens: 1_200,
        allow_private_targets: state.config.core.allow_private_providers,
    };
    let catalog = fetch_catalog(&state, request.provider_kind, &config).await?;
    Ok(Json(catalog.into()))
}

// ---------------------------------------------------------------------------
// Completions
// ---------------------------------------------------------------------------

#[utoipa::path(post, path = "/v1/completions", tag = "Completions", request_body = CompletionsRequest, responses((status = 200, description = "Completion"), (status = 400, description = "Invalid request", body = Problem)))]
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Json(request): Json<CompletionsRequest>,
) -> Result<Response, ApiError> {
    check_rate_limit(&state, &principal)?;
    let (settings, config) = resolve_provider(&state, &principal.scope).await?;
    let provider = state.providers.get(settings.provider_kind).ok_or_else(|| {
        ApiError::unavailable(
            "provider_not_configured",
            "Provider backend is not configured.",
        )
    })?;

    let latest_user = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == ChatRole::User)
        .map(|message| message.content.as_str())
        .unwrap_or_default();
    if let Some(hit) = clypeus_core::guard::classify(state.guard.as_ref(), latest_user) {
        clypeus_core::metrics::record_injection_blocked(hit.reason());
        let content = state.guard.refusal().to_string();
        if request.stream {
            let payload = clypeus_core::sse::content_delta(&content);
            let done = String::from_utf8_lossy(clypeus_core::sse::DONE_EVENT).to_string();
            let body = Body::from(format!("{payload}{done}"));
            return Ok(sse_response(body));
        }
        return Ok(Json(json!({
            "content": content,
            "reasoning": null,
            "model": request.model,
            "usage": null,
            "toolCalls": [],
        }))
        .into_response());
    }

    let reasoning = request
        .reasoning_level
        .as_deref()
        .and_then(ReasoningLevel::parse)
        .filter(|level| *level != ReasoningLevel::Default);
    let tools = request.tools.clone().unwrap_or_default();
    let tool_choice = request
        .tool_choice
        .as_deref()
        .and_then(clypeus_core::provider::ToolChoice::parse)
        .unwrap_or_default();
    let completion = clypeus_core::provider::CompletionRequest {
        model: request.model.clone(),
        messages: request.messages.clone(),
        reasoning,
        tools,
        tool_choice,
        max_output_tokens: request
            .max_output_tokens
            .unwrap_or(settings.max_output_tokens)
            .clamp(1, settings.max_output_tokens.max(1)),
    };

    if request.stream {
        let mut stream = provider.stream(&config, completion).await?;
        let mut buffer = String::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(deltas) => {
                    for delta in deltas {
                        buffer.push_str(&match delta {
                            clypeus_core::provider::StreamDelta::Content(text) => {
                                clypeus_core::sse::content_delta(&text)
                            }
                            clypeus_core::provider::StreamDelta::Reasoning(text) => {
                                clypeus_core::sse::reasoning_delta(&text)
                            }
                        });
                    }
                }
                Err(error) => {
                    buffer.push_str(&clypeus_core::sse::error(error.code()));
                    break;
                }
            }
        }
        let outcome = stream.into_outcome();
        if let Some(usage) = outcome.usage.as_ref().and_then(clypeus_core::sse::usage) {
            buffer.push_str(&usage);
        }
        buffer.push_str(&String::from_utf8_lossy(clypeus_core::sse::DONE_EVENT));
        return Ok(sse_response(Body::from(buffer)));
    }

    let outcome = provider.complete(&config, completion).await?;
    Ok(Json(json!({
        "content": outcome.content,
        "reasoning": outcome.reasoning,
        "model": outcome.model.unwrap_or(request.model),
        "usage": outcome.usage,
        "toolCalls": outcome.tool_calls,
    }))
    .into_response())
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

#[utoipa::path(get, path = "/v1/threads", tag = "Threads", responses((status = 200, description = "Threads", body = ThreadListResponse)))]
pub async fn list_threads(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<ThreadListResponse>, ApiError> {
    let threads = state
        .conversation
        .list_threads(&principal.scope, &principal.subject)
        .await?;
    Ok(Json(ThreadListResponse {
        threads: threads.into_iter().map(Into::into).collect(),
    }))
}

#[utoipa::path(post, path = "/v1/threads", tag = "Threads", request_body = CreateThreadRequest, responses((status = 200, description = "Thread", body = ThreadDto)))]
pub async fn create_thread(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Json(request): Json<CreateThreadRequest>,
) -> Result<Json<ThreadDto>, ApiError> {
    let thread = state
        .conversation
        .create_thread(
            &principal.scope,
            &principal.subject,
            CreateThread {
                title: request.title,
                model: request.model,
                pinned: request.pinned,
            },
        )
        .await?;
    Ok(Json(thread.into()))
}

#[utoipa::path(get, path = "/v1/threads/{thread_id}", tag = "Threads", params(("thread_id" = Uuid, Path, description = "Thread id")), responses((status = 200, description = "Thread view", body = ThreadViewResponse), (status = 404, description = "Not found", body = Problem)))]
pub async fn get_thread(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(thread_id): Path<Uuid>,
) -> Result<Json<ThreadViewResponse>, ApiError> {
    let view = state
        .conversation
        .thread_view(&principal.scope, &principal.subject, thread_id)
        .await?;
    Ok(Json(view.into()))
}

#[utoipa::path(patch, path = "/v1/threads/{thread_id}", tag = "Threads", params(("thread_id" = Uuid, Path, description = "Thread id")), request_body = UpdateThreadRequest, responses((status = 200, description = "Thread", body = ThreadDto)))]
pub async fn update_thread(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(thread_id): Path<Uuid>,
    Json(request): Json<UpdateThreadRequest>,
) -> Result<Json<ThreadDto>, ApiError> {
    let thread = state
        .conversation
        .update_thread(
            &principal.scope,
            &principal.subject,
            thread_id,
            ThreadUpdate {
                title: request.title,
                pinned: request.pinned,
                model: request.model,
            },
        )
        .await?;
    Ok(Json(thread.into()))
}

#[utoipa::path(delete, path = "/v1/threads/{thread_id}", tag = "Threads", params(("thread_id" = Uuid, Path, description = "Thread id")), responses((status = 204, description = "Deleted")))]
pub async fn delete_thread(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(thread_id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    state
        .conversation
        .delete_thread(&principal.scope, &principal.subject, thread_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

#[utoipa::path(get, path = "/v1/threads/{thread_id}/usage", tag = "Threads", params(("thread_id" = Uuid, Path, description = "Thread id")), responses((status = 200, description = "Usage", body = UsageResponse)))]
pub async fn thread_usage(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(thread_id): Path<Uuid>,
) -> Result<Json<UsageResponse>, ApiError> {
    let usage = state
        .conversation
        .thread_usage(&principal.scope, &principal.subject, thread_id)
        .await?;
    Ok(Json(usage.into()))
}

// ---------------------------------------------------------------------------
// Turns
// ---------------------------------------------------------------------------

#[utoipa::path(post, path = "/v1/threads/{thread_id}/messages", tag = "Turns", params(("thread_id" = Uuid, Path, description = "Thread id")), request_body = CreateTurnRequest, responses((status = 200, description = "Turn", body = TurnResponse)))]
pub async fn create_message(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(thread_id): Path<Uuid>,
    Json(request): Json<CreateTurnRequest>,
) -> Result<Response, ApiError> {
    run_turn(
        state,
        principal,
        thread_id,
        TurnPlan {
            target: StoreTurnTarget::New,
            content: request.content,
            stream: request.stream,
            model: request.model,
            reasoning_level: request.reasoning_level,
            page_context: request.page_context.map(|context| context.to_value()),
        },
    )
    .await
}

#[utoipa::path(patch, path = "/v1/messages/{message_id}", tag = "Turns", params(("message_id" = Uuid, Path, description = "User message id")), request_body = EditTurnRequest, responses((status = 200, description = "Edited turn", body = TurnResponse)))]
pub async fn edit_message(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(message_id): Path<Uuid>,
    Json(request): Json<EditTurnRequest>,
) -> Result<Response, ApiError> {
    let message = state
        .conversation
        .message(&principal.scope, &principal.subject, message_id)
        .await?;
    run_turn(
        state,
        principal,
        message.thread_id,
        TurnPlan {
            target: StoreTurnTarget::Edit { message_id },
            content: request.content,
            stream: request.stream,
            model: request.model,
            reasoning_level: request.reasoning_level,
            page_context: request.page_context.map(|context| context.to_value()),
        },
    )
    .await
}

#[utoipa::path(post, path = "/v1/messages/{message_id}/regenerate", tag = "Turns", params(("message_id" = Uuid, Path, description = "Assistant message id")), request_body = RegenerateTurnRequest, responses((status = 200, description = "Regenerated turn", body = TurnResponse)))]
pub async fn regenerate_message(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(message_id): Path<Uuid>,
    Json(request): Json<RegenerateTurnRequest>,
) -> Result<Response, ApiError> {
    let message = state
        .conversation
        .message(&principal.scope, &principal.subject, message_id)
        .await?;
    run_turn(
        state,
        principal,
        message.thread_id,
        TurnPlan {
            target: StoreTurnTarget::Regenerate { message_id },
            content: message.content,
            stream: request.stream,
            model: request.model,
            reasoning_level: request.reasoning_level,
            page_context: request.page_context.map(|context| context.to_value()),
        },
    )
    .await
}

#[utoipa::path(post, path = "/v1/messages/{message_id}/activate", tag = "Turns", params(("message_id" = Uuid, Path, description = "Message id")), responses((status = 204, description = "Activated")))]
pub async fn activate_message(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(message_id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    state
        .conversation
        .activate_message(&principal.scope, &principal.subject, message_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

#[utoipa::path(put, path = "/v1/messages/{message_id}/feedback", tag = "Turns", params(("message_id" = Uuid, Path, description = "Assistant message id")), request_body = SetFeedbackRequest, responses((status = 200, description = "Feedback", body = FeedbackDto)))]
pub async fn set_feedback(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(message_id): Path<Uuid>,
    Json(request): Json<SetFeedbackRequest>,
) -> Result<Json<FeedbackDto>, ApiError> {
    let rating = clypeus_core::store::FeedbackRating::parse(&request.rating)
        .ok_or_else(|| ApiError::bad_request("invalid_rating", "rating must be 'up' or 'down'."))?;
    let feedback = state
        .conversation
        .set_feedback(
            &principal.scope,
            &principal.subject,
            message_id,
            rating,
            request.comment.as_deref(),
        )
        .await?;
    Ok(Json(feedback.into()))
}

#[utoipa::path(delete, path = "/v1/messages/{message_id}/feedback", tag = "Turns", params(("message_id" = Uuid, Path, description = "Assistant message id")), responses((status = 204, description = "Cleared")))]
pub async fn clear_feedback(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(message_id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    state
        .conversation
        .clear_feedback(&principal.scope, &principal.subject, message_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

struct TurnPlan {
    target: StoreTurnTarget,
    content: String,
    stream: bool,
    model: Option<String>,
    reasoning_level: Option<String>,
    page_context: Option<Value>,
}

async fn run_turn(
    state: Arc<AppState>,
    principal: Principal,
    thread_id: Uuid,
    plan: TurnPlan,
) -> Result<Response, ApiError> {
    check_rate_limit(&state, &principal)?;
    let (settings, config) = resolve_provider(&state, &principal.scope).await?;
    let catalog = fetch_catalog(&state, settings.provider_kind, &config).await?;
    let (model, reasoning) = clypeus_core::functions::select_model(
        &catalog,
        settings.default_model.as_deref(),
        plan.model.as_deref(),
        plan.reasoning_level.as_deref(),
    )
    .map_err(|error| ApiError::bad_request(error.code, error.detail))?;
    let reasoning = reasoning.and_then(|level| ReasoningLevel::parse(&level));

    let context = turn_context(plan.page_context.clone());
    let context_json = serde_json::to_value(&context).ok();
    let profile_prompt = profile_prompt(&settings.profile);
    let system = clypeus_core::context::build_system_message(profile_prompt.as_deref(), &context);

    let started = state
        .conversation
        .begin_turn(
            &principal.scope,
            &principal.subject,
            clypeus_core::store::BeginTurn {
                thread_id: Some(thread_id),
                target: plan.target,
                content: plan.content,
                title: None,
                model: Some(model.clone()),
                reasoning_level: reasoning.map(|level| level.as_wire().to_string()),
                context_version: Some(context.version.clone()),
                context_json,
            },
        )
        .await?;

    let mut messages = state
        .conversation
        .history_for_message(
            &principal.scope,
            &principal.subject,
            started.assistant_message.id,
        )
        .await?;
    if let Some(system) = system {
        messages.insert(0, ChatMessage::text(ChatRole::System, system));
    }

    let tools = state.broker.specs_for(&principal.scopes);
    let limits = state.turn_limits();
    let caller = Caller::new(
        principal.clone(),
        Uuid::new_v4().simple().to_string(),
        thread_id,
        started.assistant_message.id,
        budget_for(limits),
    );
    let turn = TurnRequest {
        provider_kind: settings.provider_kind,
        provider: config,
        model,
        reasoning,
        messages,
        tools,
        caller,
        user_message_id: started.user_message.id,
        assistant_message_id: started.assistant_message.id,
        context,
        limits,
    };

    if plan.stream {
        let stream = Arc::clone(&state.orchestrator).spawn_streamed(turn);
        return Ok(sse_response_stream(stream));
    }

    let outcome = state
        .orchestrator
        .run_buffered(turn)
        .await
        .map_err(ApiError::from)?;
    let thread = state
        .conversation
        .get_thread(&principal.scope, &principal.subject, thread_id)
        .await?
        .ok_or_else(|| ApiError::not_found("not_found", "The thread was not found."))?;
    let user_message = state
        .conversation
        .message(
            &principal.scope,
            &principal.subject,
            started.user_message.id,
        )
        .await?;
    let assistant_message = state
        .conversation
        .message(
            &principal.scope,
            &principal.subject,
            started.assistant_message.id,
        )
        .await?;
    let _ = outcome;
    Ok(Json(TurnResponse {
        thread: thread.into(),
        user_message: user_message.into(),
        assistant_message: assistant_message.into(),
    })
    .into_response())
}

fn turn_context(page_context: Option<Value>) -> TurnContext {
    match page_context {
        Some(Value::Object(fields)) if !fields.is_empty() => TurnContext::from_fields(fields),
        _ => TurnContext::empty(),
    }
}

/// Renders the configured profile selection. The standalone server ships no
/// built-in domain profile, so `Builtin` contributes only custom text.
fn profile_prompt(selection: &ProfileSelection) -> Option<String> {
    match selection {
        ProfileSelection::Builtin { custom } => custom
            .as_deref()
            .map(str::trim)
            .filter(|custom| !custom.is_empty())
            .map(str::to_string),
        ProfileSelection::Custom(prompt) => {
            let prompt = prompt.trim();
            (!prompt.is_empty()).then(|| prompt.to_string())
        }
        ProfileSelection::Disabled => None,
    }
}

// ---------------------------------------------------------------------------
// Approvals
// ---------------------------------------------------------------------------

#[utoipa::path(post, path = "/v1/tool-calls/{tool_call_id}/approvals", tag = "Approvals", params(("tool_call_id" = String, Path, description = "Tool call id")), request_body = SubmitApprovalRequest, responses((status = 200, description = "Resumed turn", body = TurnResponse), (status = 409, description = "Expired or replayed", body = Problem)))]
pub async fn submit_approval(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(tool_call_id): Path<String>,
    Json(request): Json<SubmitApprovalRequest>,
) -> Result<Response, ApiError> {
    let snapshot = state
        .broker
        .load_decision_target(&principal.scope, &principal.subject, &tool_call_id)
        .await
        .map_err(ApiError::from)?;

    if snapshot.status != ToolCallStatus::AwaitingApproval.as_wire() {
        return Err(ApiError::conflict(
            "tool_approval_replayed",
            "This call is not awaiting a decision.",
        ));
    }
    if snapshot
        .expires_at
        .is_some_and(|expires| expires < chrono::Utc::now())
    {
        let _ = state.approvals.expire_tool_calls(chrono::Utc::now()).await;
        return Err(ApiError::conflict(
            "tool_approval_expired",
            "The approval window expired.",
        ));
    }
    if snapshot.arguments_hash != request.arguments_hash {
        return Err(ApiError::conflict(
            "tool_arguments_changed",
            "The arguments hash does not match the parked call.",
        ));
    }
    if snapshot.approval_kind.as_deref() == Some("typed_confirm") {
        let Some(field) = snapshot.confirm_field.as_deref() else {
            return Err(ApiError::internal("Approval is misconfigured."));
        };
        let expected = snapshot
            .arguments
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or_default();
        if request.typed_confirm.as_deref() != Some(expected) {
            return Err(ApiError::bad_request(
                "typed_confirm_mismatch",
                "The typed confirmation does not match the target.",
            ));
        }
    }

    let decision = request.decision.trim().to_ascii_lowercase();
    if decision != "allow" && decision != "deny" {
        return Err(ApiError::bad_request(
            "invalid_decision",
            "decision must be 'allow' or 'deny'.",
        ));
    }

    let tool_request = ToolCallRequest {
        id: snapshot.id.clone(),
        name: snapshot.name.clone(),
        arguments: snapshot.arguments.clone(),
    };
    let message = state
        .conversation
        .message(&principal.scope, &principal.subject, snapshot.message_id)
        .await?;
    let thread_id = message.thread_id;

    let approval_id = Uuid::new_v4().to_string();
    if decision == "deny" {
        state.approvals.mark_tool_call_denied(&snapshot.id).await?;
        let tool_outcome = state
            .broker
            .deny(
                &Caller::new(
                    principal.clone(),
                    Uuid::new_v4().simple().to_string(),
                    thread_id,
                    snapshot.message_id,
                    budget_for(state.turn_limits()),
                ),
                &tool_request,
            )
            .await;
        let _ = tool_outcome;
    } else {
        state
            .approvals
            .insert_tool_approval(clypeus_core::store::NewToolApproval {
                id: &approval_id,
                tool_call_id: &snapshot.id,
                scope: &principal.scope,
                subject: &principal.subject,
                decision: "allow",
                reason: request.reason.as_deref(),
                typed_confirm: request.typed_confirm.as_deref(),
                arguments_hash: &snapshot.arguments_hash,
                expires_at: snapshot.expires_at.unwrap_or_else(chrono::Utc::now),
            })
            .await?;
        state
            .approvals
            .mark_tool_call_approved(&snapshot.id, &approval_id)
            .await?;
    }

    let (settings, config) = resolve_provider(&state, &principal.scope).await?;
    let context = match message.context_json.clone() {
        Some(Value::Object(fields)) => TurnContext::from_fields(fields),
        _ => TurnContext::empty(),
    };
    let caller = Caller::new(
        principal.clone(),
        Uuid::new_v4().simple().to_string(),
        thread_id,
        snapshot.message_id,
        budget_for(state.turn_limits()),
    );
    let action = if decision == "allow" {
        ResumeAction::Approved {
            request: tool_request,
            grant: ApprovalGrant {
                approval_id,
                arguments_hash: snapshot.arguments_hash.clone(),
            },
        }
    } else {
        ResumeAction::Denied {
            request: tool_request,
        }
    };
    let resume = ResumeRequest {
        provider: ResumeProvider {
            provider_kind: settings.provider_kind,
            provider: config,
            model: message.model.clone().unwrap_or_default(),
            reasoning: message
                .reasoning_level
                .as_deref()
                .and_then(ReasoningLevel::parse),
        },
        tools: state.broker.specs_for(&principal.scopes),
        caller,
        assistant_message_id: snapshot.message_id,
        user_message_id: match state
            .conversation
            .user_anchor_message(&principal.scope, &principal.subject, snapshot.message_id)
            .await?
        {
            Some(anchor) => anchor,
            None => snapshot.message_id,
        },
        context,
        limits: state.turn_limits(),
        action,
    };

    if request.stream {
        let stream = Arc::clone(&state.orchestrator).spawn_resumed_streamed(resume);
        return Ok(sse_response_stream(stream));
    }
    state
        .orchestrator
        .run_resumed_buffered(resume)
        .await
        .map_err(ApiError::from)?;
    let message = state
        .conversation
        .message(&principal.scope, &principal.subject, snapshot.message_id)
        .await?;
    let thread = state
        .conversation
        .get_thread(&principal.scope, &principal.subject, thread_id)
        .await?
        .ok_or_else(|| ApiError::not_found("not_found", "The thread was not found."))?;
    let user_message = state
        .conversation
        .message(
            &principal.scope,
            &principal.subject,
            message.parent_message_id.unwrap_or(message.id),
        )
        .await
        .unwrap_or_else(|_| message.clone());
    Ok(Json(TurnResponse {
        thread: thread.into(),
        user_message: user_message.into(),
        assistant_message: message.into(),
    })
    .into_response())
}

// ---------------------------------------------------------------------------
// Tools and functions
// ---------------------------------------------------------------------------

#[utoipa::path(get, path = "/v1/tools", tag = "Tools", responses((status = 200, description = "Tool catalog", body = ToolCatalogResponse)))]
pub async fn list_tools(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<ToolCatalogResponse>, ApiError> {
    let catalog = state.broker.catalog_for(&principal.scopes);
    Ok(Json(ToolCatalogResponse {
        tools: catalog
            .into_iter()
            .map(|tool| ToolCatalogEntryDto {
                name: tool.name,
                description: tool.description,
                schema: tool.input_schema,
                required_scopes: tool.required_scopes,
                risk: tool.risk.as_str().to_string(),
                approval: tool.approval.as_str().to_string(),
                confirm_field: tool.confirm_field,
            })
            .collect(),
    }))
}

#[utoipa::path(get, path = "/v1/functions", tag = "Functions", responses((status = 200, description = "Function catalog", body = clypeus_core::functions::FunctionListResponse)))]
pub async fn list_functions(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<clypeus_core::functions::FunctionListResponse>, ApiError> {
    Ok(Json(clypeus_core::functions::FunctionListResponse {
        functions: state.functions.descriptors_for(&principal.scopes),
    }))
}

#[utoipa::path(post, path = "/v1/functions/{name}", tag = "Functions", params(("name" = String, Path, description = "Function name")), request_body = RunFunctionRequest, responses((status = 200, description = "Function result", body = clypeus_core::functions::RunFunctionResponse)))]
pub async fn run_function(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(name): Path<String>,
    Json(request): Json<RunFunctionRequest>,
) -> Result<Json<clypeus_core::functions::RunFunctionResponse>, ApiError> {
    let function = state.functions.get(&name).ok_or_else(|| {
        ApiError::not_found("function_not_found", "The function is not registered.")
    })?;
    if !function
        .required_scopes()
        .iter()
        .all(|required| principal.has_scope(required))
    {
        return Err(ApiError::forbidden(
            "function_not_permitted",
            "The function is not available to this caller.",
        ));
    }
    let response = state
        .function_runner
        .run(&principal, function, request)
        .await
        .map_err(|error| match error.kind {
            clypeus_core::functions::FunctionRunErrorKind::Invalid => {
                ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, error.code, error.detail)
            }
            clypeus_core::functions::FunctionRunErrorKind::RateLimited => {
                ApiError::new(StatusCode::TOO_MANY_REQUESTS, error.code, error.detail)
            }
            clypeus_core::functions::FunctionRunErrorKind::Unavailable => {
                ApiError::unavailable(error.code, error.detail)
            }
            clypeus_core::functions::FunctionRunErrorKind::Upstream => {
                ApiError::new(StatusCode::BAD_GATEWAY, error.code, error.detail)
            }
        })?;
    Ok(Json(response))
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuditQueryParams {
    #[serde(default)]
    pub limit: Option<i64>,
    #[serde(default)]
    pub offset: Option<i64>,
    #[serde(default)]
    pub item_name: Option<String>,
    #[serde(default)]
    pub outcome: Option<String>,
}

#[utoipa::path(get, path = "/v1/audit", tag = "Audit", params(("limit" = Option<i64>, Query, description = "Page size"), ("offset" = Option<i64>, Query, description = "Page offset")), responses((status = 200, description = "Audit page", body = AuditListResponse)))]
pub async fn list_audit(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Query(params): Query<AuditQueryParams>,
) -> Result<Json<AuditListResponse>, ApiError> {
    require_admin(&state, &principal)?;
    let page = state
        .audit_reader
        .page(
            &principal.scope,
            AuditQuery {
                limit: params.limit.unwrap_or(50),
                offset: params.offset.unwrap_or(0),
                item_name: params.item_name,
                outcome: params.outcome,
                ..AuditQuery::default()
            },
        )
        .await?;
    Ok(Json(AuditListResponse {
        entries: page.entries,
        limit: page.limit,
        offset: page.offset,
        total: page.total,
    }))
}

#[utoipa::path(get, path = "/v1/audit/export", tag = "Audit", responses((status = 200, description = "Audit export")))]
pub async fn export_audit(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<Vec<clypeus_core::audit::AuditRecord>>, ApiError> {
    require_admin(&state, &principal)?;
    let entries = state
        .audit_reader
        .export(&principal.scope, AuditQuery::default())
        .await?;
    Ok(Json(entries))
}

// ---------------------------------------------------------------------------
// Admin settings
// ---------------------------------------------------------------------------

#[utoipa::path(get, path = "/admin/v1/scopes/{scope}/settings", tag = "Admin", params(("scope" = String, Path, description = "Scope id")), responses((status = 200, description = "Scope settings", body = ScopeSettingsDto)))]
pub async fn get_scope_settings(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(scope): Path<String>,
) -> Result<Json<ScopeSettingsDto>, ApiError> {
    require_admin(&state, &principal)?;
    let scope = ScopeId::new(scope);
    let settings = state.settings.get(&scope).await?.ok_or_else(|| {
        ApiError::not_found("settings_not_found", "No settings exist for this scope.")
    })?;
    let mut dto: ScopeSettingsDto = settings.into();
    dto.scope_id = scope.to_string();
    Ok(Json(dto))
}

#[utoipa::path(put, path = "/admin/v1/scopes/{scope}/settings", tag = "Admin", params(("scope" = String, Path, description = "Scope id")), request_body = UpdateScopeSettingsRequest, responses((status = 200, description = "Scope settings", body = ScopeSettingsDto)))]
pub async fn put_scope_settings(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(scope): Path<String>,
    Json(request): Json<UpdateScopeSettingsRequest>,
) -> Result<Json<ScopeSettingsDto>, ApiError> {
    require_admin(&state, &principal)?;
    let scope = ScopeId::new(scope);
    if let Some(base_url) = request
        .base_url
        .as_deref()
        .filter(|url| !url.trim().is_empty())
    {
        validate_base_url(base_url, state.config.core.allow_private_providers)?;
    }
    let profile = match request.profile.clone() {
        Some(value) => Some(
            serde_json::from_value::<ProfileSelection>(value)
                .map_err(|error| ApiError::bad_request("invalid_profile", error.to_string()))?,
        ),
        None => None,
    };
    let api_key_present = if let Some(api_key) = request.api_key.clone() {
        if api_key.trim().is_empty() {
            None
        } else {
            state
                .secrets
                .put(
                    &scope,
                    clypeus_core::functions::PROVIDER_API_KEY,
                    clypeus_core::secrets::SecretString::new(api_key),
                )
                .await
                .map_err(|_| ApiError::internal("The API key could not be stored."))?;
            Some(true)
        }
    } else if request.clear_api_key {
        let _ = state
            .secrets
            .delete(&scope, clypeus_core::functions::PROVIDER_API_KEY)
            .await;
        Some(false)
    } else {
        None
    };
    let settings = state
        .settings
        .upsert(
            &scope,
            ScopeSettingsUpdate {
                provider_kind: request.provider_kind,
                base_url: request.base_url.clone(),
                default_model: request.default_model.clone(),
                timeout_ms: request.timeout_ms,
                max_output_tokens: request.max_output_tokens,
                api_key_present,
                profile,
                extensions: Some(json!({"scopeId": scope.to_string()})),
            },
        )
        .await?;
    let mut dto: ScopeSettingsDto = settings.into();
    dto.scope_id = scope.to_string();
    Ok(Json(dto))
}

#[utoipa::path(post, path = "/admin/v1/scopes/{scope}/settings/test", tag = "Admin", params(("scope" = String, Path, description = "Scope id")), responses((status = 200, description = "Probe result", body = ConnectionTestDto)))]
pub async fn test_scope_settings(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(scope): Path<String>,
) -> Result<Json<ConnectionTestDto>, ApiError> {
    require_admin(&state, &principal)?;
    let scope = ScopeId::new(scope);
    let (settings, config) = resolve_provider(&state, &scope).await?;
    let provider = state.providers.get(settings.provider_kind).ok_or_else(|| {
        ApiError::unavailable(
            "provider_not_configured",
            "Provider backend is not configured.",
        )
    })?;
    let report = provider.probe(&config).await;
    Ok(Json(report.into()))
}

#[utoipa::path(post, path = "/admin/v1/scopes/{scope}/models/preview", tag = "Admin", params(("scope" = String, Path, description = "Scope id")), request_body = ModelsPreviewRequest, responses((status = 200, description = "Model catalog", body = ModelsResponse)))]
pub async fn admin_preview_models(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    Path(_scope): Path<String>,
    Json(request): Json<ModelsPreviewRequest>,
) -> Result<Json<ModelsResponse>, ApiError> {
    require_admin(&state, &principal)?;
    validate_base_url(&request.base_url, state.config.core.allow_private_providers)?;
    let config = ProviderConfig {
        base_url: request.base_url,
        api_key: request.api_key.into(),
        timeout: Duration::from_secs(30),
        max_output_tokens: 1_200,
        allow_private_targets: state.config.core.allow_private_providers,
    };
    let catalog = fetch_catalog(&state, request.provider_kind, &config).await?;
    Ok(Json(catalog.into()))
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

fn sse_content_type() -> [(header::HeaderName, &'static str); 2] {
    [
        (header::CONTENT_TYPE, "text/event-stream"),
        (header::CACHE_CONTROL, "no-cache"),
    ]
}

fn sse_response(body: Body) -> Response {
    let mut response = Response::new(body);
    for (name, value) in sse_content_type() {
        response.headers_mut().insert(name, value.parse().unwrap());
    }
    response
}

fn sse_response_stream(
    stream: futures_util::stream::BoxStream<'static, Result<bytes::Bytes, std::io::Error>>,
) -> Response {
    sse_response(Body::from_stream(stream))
}

/// Marker used by the OpenAPI module to ensure the tool error taxonomy is
/// documented.
pub const _TOOL_ERROR_CODES: &[&str] = &[
    "tool_unknown",
    "tool_not_permitted",
    "tool_invalid_arguments",
    "tool_timeout",
    "tool_upstream_unavailable",
    "tool_approval_required",
    "tool_approval_denied",
    "tool_approval_expired",
    "tool_arguments_changed",
    "tool_approval_replayed",
    "tool_limit_exceeded",
    "tool_rate_limited",
    "tool_result_too_large",
    "tool_internal",
];

/// Suppresses unused import warnings for taxonomy items referenced only in
/// generated documentation.
#[allow(unused)]
fn _taxonomy() -> Option<(ToolError, ProviderKind, ToolSpec, MessageStatus)> {
    None
}
