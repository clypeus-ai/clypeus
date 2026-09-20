//! HTTP error mapping. Every error leaves the server as a problem document
//! with a stable `code`.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::json;

use clypeus_core::principal::AuthError;
use clypeus_core::provider::ProviderError;
use clypeus_core::store::StoreError;
use clypeus_core::tools::ToolError;

/// Problem response body.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Problem {
    pub r#type: String,
    pub title: String,
    pub status: u16,
    pub detail: String,
    pub code: String,
}

/// Error carrying an HTTP status and a stable code.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: String,
    pub detail: String,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for ApiError {}

impl ApiError {
    pub fn new(status: StatusCode, code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            detail: detail.into(),
        }
    }

    pub fn bad_request(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, code, detail)
    }

    pub fn not_found(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, code, detail)
    }

    pub fn conflict(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, code, detail)
    }

    pub fn forbidden(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, code, detail)
    }

    pub fn unavailable(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, code, detail)
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", detail)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Problem {
            r#type: "about:blank".to_string(),
            title: self
                .status
                .canonical_reason()
                .unwrap_or("error")
                .to_string(),
            status: self.status.as_u16(),
            detail: self.detail,
            code: self.code,
        };
        (self.status, Json(body)).into_response()
    }
}

impl From<AuthError> for ApiError {
    fn from(error: AuthError) -> Self {
        let status = match error.code {
            "missing_credentials" => StatusCode::UNAUTHORIZED,
            _ => StatusCode::UNAUTHORIZED,
        };
        Self::new(status, error.code, "Authentication failed.")
    }
}

impl From<StoreError> for ApiError {
    fn from(error: StoreError) -> Self {
        match error {
            StoreError::NotFound => Self::not_found("not_found", "The resource was not found."),
            StoreError::Validation(detail) => Self::bad_request("validation_error", detail),
            StoreError::Conflict(detail) => Self::conflict("conflict", detail),
            StoreError::Backend(detail) => {
                tracing::error!(%detail, "store error");
                Self::internal("The request could not be completed.")
            }
        }
    }
}

impl From<ProviderError> for ApiError {
    fn from(error: ProviderError) -> Self {
        match &error {
            ProviderError::InvalidBaseUrl(detail) => {
                Self::bad_request("provider_invalid_base_url", detail.clone())
            }
            ProviderError::Timeout => Self::new(
                StatusCode::GATEWAY_TIMEOUT,
                "provider_timeout",
                "The provider timed out.",
            ),
            ProviderError::Upstream { status, .. } if *status == 429 => Self::new(
                StatusCode::TOO_MANY_REQUESTS,
                "provider_unavailable",
                "The provider is rate limiting requests; try again later.",
            ),
            _ => {
                tracing::warn!(detail = %error, "provider failure");
                Self::new(StatusCode::BAD_GATEWAY, error.code(), error.safe_message())
            }
        }
    }
}

impl From<ToolError> for ApiError {
    fn from(error: ToolError) -> Self {
        match error {
            ToolError::NotFound => Self::not_found("tool_not_found", "The resource was not found."),
            ToolError::ApprovalExpired => {
                Self::conflict("tool_approval_expired", "The approval window expired.")
            }
            ToolError::ApprovalReplayed => Self::conflict(
                "tool_approval_replayed",
                "This approval was already consumed.",
            ),
            ToolError::ArgumentsChanged => Self::conflict(
                "tool_arguments_changed",
                "The arguments changed after approval was granted.",
            ),
            ToolError::NotPermitted => Self::forbidden(
                "tool_not_permitted",
                "The tool is not available to this caller.",
            ),
            other => Self::bad_request(other.code(), other.message()),
        }
    }
}

/// Convenience response for the probes.
pub fn probe_body(status: &str, detail: Option<&str>) -> Json<serde_json::Value> {
    Json(json!({
        "status": status,
        "detail": detail,
    }))
}
