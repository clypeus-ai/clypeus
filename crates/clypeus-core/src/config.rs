//! Core configuration read from `CLYPEUS_*` environment variables.

use std::time::Duration;

use crate::orchestrator::TurnLimits;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

/// Process-wide limits and defaults.
#[derive(Debug, Clone)]
pub struct CoreConfig {
    pub limits: TurnLimits,
    /// Chat requests per subject per window.
    pub chat_rate_limit: u32,
    pub chat_rate_window: Duration,
    /// Function runs per subject per window.
    pub function_rate_limit: u32,
    /// Approval window for parked write calls.
    pub approval_ttl: Duration,
    /// Non-denied write calls per `(scope, subject, tool)` per 24h. `None`
    /// disables the quota.
    pub write_quota_per_day: Option<u32>,
    /// Audit retention in days; `0` disables purging.
    pub audit_retention_days: u64,
    /// Allows provider base URLs that resolve to private addresses.
    pub allow_private_providers: bool,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            limits: TurnLimits::default(),
            chat_rate_limit: 60,
            chat_rate_window: Duration::from_secs(15 * 60),
            function_rate_limit: 30,
            approval_ttl: crate::broker::APPROVAL_TTL,
            write_quota_per_day: None,
            audit_retention_days: 90,
            allow_private_providers: false,
        }
    }
}

impl CoreConfig {
    /// Reads configuration from the environment. Invalid values fall back to
    /// the defaults; out-of-range values are clamped.
    pub fn from_env() -> Self {
        let defaults = Self::default();
        let mut limits = TurnLimits::default();
        limits.max_tool_rounds =
            env_usize("CLYPEUS_MAX_TOOL_ROUNDS", limits.max_tool_rounds).clamp(1, 16);
        limits.max_tool_calls =
            env_usize("CLYPEUS_MAX_TOOL_CALLS", limits.max_tool_calls).clamp(1, 64);
        limits.max_result_bytes =
            env_usize("CLYPEUS_MAX_TURN_RESULT_BYTES", limits.max_result_bytes)
                .clamp(1024, 4 * 1024 * 1024);
        limits.turn_budget = Duration::from_secs(
            env_u64("CLYPEUS_TURN_BUDGET_SECONDS", limits.turn_budget.as_secs()).clamp(1, 3600),
        );
        Self {
            limits,
            chat_rate_limit: env_u32("CLYPEUS_CHAT_RATE_LIMIT", defaults.chat_rate_limit)
                .clamp(1, 100_000),
            chat_rate_window: Duration::from_secs(
                env_u64(
                    "CLYPEUS_CHAT_RATE_WINDOW_SECONDS",
                    defaults.chat_rate_window.as_secs(),
                )
                .clamp(1, 86_400),
            ),
            function_rate_limit: env_u32(
                "CLYPEUS_FUNCTION_RATE_LIMIT",
                defaults.function_rate_limit,
            )
            .clamp(1, 100_000),
            approval_ttl: Duration::from_secs(
                env_u64(
                    "CLYPEUS_APPROVAL_TTL_SECONDS",
                    defaults.approval_ttl.as_secs(),
                )
                .clamp(10, 86_400),
            ),
            write_quota_per_day: match std::env::var("CLYPEUS_WRITE_QUOTA_PER_DAY") {
                Ok(value) if value.trim() == "0" => None,
                Ok(value) => value
                    .trim()
                    .parse()
                    .ok()
                    .map(|limit: u32| limit.clamp(1, 100_000)),
                Err(_) => defaults.write_quota_per_day,
            },
            audit_retention_days: env_u64(
                "CLYPEUS_AUDIT_RETENTION_DAYS",
                defaults.audit_retention_days,
            )
            .min(3650),
            allow_private_providers: env_bool(
                "CLYPEUS_ALLOW_PRIVATE_PROVIDERS",
                defaults.allow_private_providers,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_bounded() {
        let config = CoreConfig::default();
        assert!(config.limits.max_tool_rounds > 0);
        assert!(config.limits.turn_budget.as_secs() > 0);
        assert!(config.chat_rate_window.as_secs() > 0);
    }
}
