//! Prometheus metrics for the tool pipeline.
//!
//! Recorder installation happens in the server; the helpers here are no-ops
//! until a recorder is installed, which keeps unit tests free of global state.

use std::time::Duration;

use metrics::{Unit, counter, describe_counter, describe_histogram, gauge, histogram};

pub fn init() {
    // A constant series so `/metrics` is never empty on a fresh process.
    gauge!("clypeus_build_info", "version" => env!("CARGO_PKG_VERSION")).set(1.0);
    describe_counter!(
        "clypeus_tool_calls_total",
        Unit::Count,
        "Tool calls by tool name and outcome"
    );
    describe_histogram!(
        "clypeus_tool_call_duration_seconds",
        Unit::Seconds,
        "Tool call duration by tool name"
    );
    describe_counter!(
        "clypeus_tool_rounds_total",
        Unit::Count,
        "Turns by number of provider tool rounds"
    );
    describe_counter!(
        "clypeus_tool_approvals_total",
        Unit::Count,
        "Tool approval decisions"
    );
    describe_counter!(
        "clypeus_injection_blocked_total",
        Unit::Count,
        "Requests refused by an injection or prompt-disclosure guard"
    );
    describe_counter!(
        "clypeus_tool_loop_blocked_total",
        Unit::Count,
        "Tool calls refused by the repeated-call loop guard"
    );
    describe_counter!(
        "clypeus_function_runs_total",
        Unit::Count,
        "AI function runs by function name and outcome"
    );
    describe_counter!(
        "clypeus_provider_requests_total",
        Unit::Count,
        "Provider requests by provider and outcome"
    );
}

/// Counts one request or call refused by a deterministic security guard.
/// `reason` is a small fixed vocabulary (`prompt_disclosure`,
/// `instruction_override`, `path_param`, `control_chars`, ...).
pub fn record_injection_blocked(reason: &str) {
    counter!(
        "clypeus_injection_blocked_total",
        "reason" => reason.to_string()
    )
    .increment(1);
}

pub fn record_tool_loop_blocked(tool: &str) {
    counter!("clypeus_tool_loop_blocked_total", "tool" => tool.to_string()).increment(1);
}

pub fn record_tool_call(tool: &str, outcome: &str, duration: Duration) {
    counter!(
        "clypeus_tool_calls_total",
        "tool" => tool.to_string(),
        "outcome" => outcome.to_string()
    )
    .increment(1);
    histogram!(
        "clypeus_tool_call_duration_seconds",
        "tool" => tool.to_string()
    )
    .record(duration.as_secs_f64());
}

pub fn record_tool_rounds(rounds: u64) {
    counter!("clypeus_tool_rounds_total", "rounds" => rounds.to_string()).increment(1);
}

pub fn record_tool_approval(decision: &str) {
    counter!(
        "clypeus_tool_approvals_total",
        "decision" => decision.to_string()
    )
    .increment(1);
}

pub fn record_function_run(function: &str, outcome: &str) {
    counter!(
        "clypeus_function_runs_total",
        "function" => function.to_string(),
        "outcome" => outcome.to_string()
    )
    .increment(1);
}

pub fn record_provider_request(provider: &str, outcome: &str) {
    counter!(
        "clypeus_provider_requests_total",
        "provider" => provider.to_string(),
        "outcome" => outcome.to_string()
    )
    .increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_render_with_clypeus_prefix() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            record_injection_blocked("prompt_disclosure");
            record_tool_loop_blocked("demo");
            record_tool_call("demo", "tool_limit_exceeded", Duration::from_millis(5));
            record_function_run("demo_fn", "succeeded");
            record_provider_request("openai", "succeeded");
        });
        let rendered = handle.render();
        assert!(
            rendered.contains("clypeus_injection_blocked_total{reason=\"prompt_disclosure\"} 1"),
            "rendered: {rendered}"
        );
        assert!(rendered.contains("clypeus_tool_loop_blocked_total{tool=\"demo\"} 1"));
        assert!(rendered.contains("clypeus_function_runs_total"));
        assert!(rendered.contains("clypeus_provider_requests_total"));
    }
}
