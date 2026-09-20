//! Core library of the Clypeus AI gateway.
//!
//! Clypeus is a policy-first gateway between an application and one or more
//! model providers. It owns four guarantees:
//!
//! * **Isolation** — every stored record carries a [`ScopeId`] and every query
//!   is filtered by the caller's scope. The scope is opaque: the core never
//!   interprets it, only partitions by it.
//! * **Typed egress** — tools cannot build URLs. A tool declares a fixed
//!   [`Egress`] target from an allowlist and the broker resolves it.
//! * **Approvals** — write and destructive tools park before execution and run
//!   only against a hash-bound, expiring, one-shot grant.
//! * **Untrusted tool data** — every tool result returned to a model is
//!   wrapped with an explicit untrusted marker.
//!
//! The crate is a library first. `clypeus-server` is a thin standalone HTTP
//! surface on top of it; applications can also embed the core directly and
//! plug their own principal resolver, policy engine, tools, functions and
//! stores.

pub mod audit;
pub mod broker;
pub mod config;
pub mod context;
pub mod functions;
pub mod guard;
pub mod metrics;
pub mod models;
pub mod orchestrator;
pub mod principal;
pub mod profile;
pub mod provider;
pub mod rate_limit;
pub mod secrets;
pub mod sse;
pub mod store;
pub mod tools;

pub use audit::{AuditItemKind, AuditPage, AuditQuery, AuditReader, AuditRecord, AuditSink};
pub use broker::{
    ApprovalChallenge, ApprovalGrant, BROKER_TOKEN_TTL, Caller, EgressAllowlist, HttpEgress,
    TokenMinter, ToolBroker, ToolCallOutcome, ToolCallRequest, TurnToolBudget, UserTokenExchanger,
    arguments_hash,
};
pub use config::CoreConfig;
pub use context::{
    EmptyTurnContextProvider, TurnContext, TurnContextInput, TurnContextProvider,
    build_system_message,
};
pub use functions::{
    AiFunction, FunctionDescriptorDto, FunctionError, FunctionListResponse, FunctionRegistry,
    FunctionRunError, FunctionRunErrorKind, FunctionRunner, PROVIDER_API_KEY, RunFunctionRequest,
    RunFunctionResponse,
};
pub use guard::{GuardHit, GuardPolicy, NeutralGuardPolicy, classify, classify_neutral};
pub use models::{
    ChatMessage, ChatRole, ProviderKind, ReasoningLevel, TokenUsage, ToolCall, ToolSpec,
};
pub use orchestrator::{
    Orchestrator, ResumeAction, ResumeProvider, ResumeRequest, TurnCompletion, TurnLimits,
    TurnOutcome, TurnRequest, budget_for,
};
pub use principal::{
    AllowAllPolicy, AuthError, AuthRequest, Decision, PolicyEngine, Principal, PrincipalResolver,
    Requirement, ResolverChain, ScopeId, StaticPolicy, StaticTokenResolver,
};
pub use profile::{ProfileSelection, ProfileStore, PromptProfile, StaticProfileStore};
pub use provider::{
    AssistantOutcome, CompletionRequest, ModelCapability, ModelCatalog, ProbeReport, Provider,
    ProviderConfig, ProviderError, ProviderRegistry, ProviderStream, SseSplitter,
    StreamAccumulator, StreamDecoder, StreamDelta, ToolChoice, validate_base_url,
};
pub use rate_limit::{InMemoryRateLimiter, RateKey, RateLimitError, RateLimiter, RateSnapshot};
pub use secrets::{SecretError, SecretStore, SecretString};
pub use store::{
    ApprovalStore, ApprovalView, AssistantFinish, BeginTurn, ConversationStore, CreateThread,
    Feedback, FeedbackRating, Message, MessageStatus, NewToolApproval, NewToolCall,
    PersistedToolCall, Readiness, ScopeSettings, ScopeSettingsStore, ScopeSettingsUpdate,
    StartedTurn, StaticScopeSettings, StoreError, Thread, ThreadUpdate, ThreadView,
    ToolCallCompletion, ToolCallSnapshot, ToolCallStatus, TurnTarget, TurnUsage,
};
pub use tools::{
    Approval, DEFAULT_TOOL_TIMEOUT, Egress, EgressAuth, EgressResponse, Risk, Tool, ToolDefinition,
    ToolEgress, ToolError, ToolExecContext, ToolLimits, ToolRegistry, wrap_untrusted_tool_result,
};
