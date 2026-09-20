//! In-memory token-bucket rate limiting keyed by scope and subject.
//!
//! The built-in limiter is process-local. Multi-replica deployments plug a
//! shared implementation behind the same [`RateLimiter`] trait.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::principal::ScopeId;

/// Key a rate limit is tracked under.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RateKey {
    pub scope: ScopeId,
    pub subject: String,
    pub namespace: &'static str,
}

impl RateKey {
    pub fn new(scope: ScopeId, subject: impl Into<String>, namespace: &'static str) -> Self {
        Self {
            scope,
            subject: subject.into(),
            namespace,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RateLimitError {
    #[error("Rate limit exceeded. Retry after {retry_after_seconds}s.")]
    Exceeded {
        remaining: u32,
        retry_after_seconds: u64,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct RateLimitConfig {
    pub max_requests: u32,
    pub window: Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 60,
            window: Duration::from_secs(15 * 60),
        }
    }
}

/// A non-consuming view of a caller's headroom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateSnapshot {
    pub limit: u32,
    pub remaining: u32,
    pub window_seconds: u64,
    pub retry_after_seconds: u64,
}

/// Admission control for one namespace.
pub trait RateLimiter: Send + Sync {
    /// Consumes one token when allowed.
    fn check(&self, key: &RateKey) -> Result<RateSnapshot, RateLimitError>;
    /// Reports headroom without consuming.
    fn peek(&self, key: &RateKey) -> RateSnapshot;
}

#[derive(Debug)]
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

/// Token-bucket limiter with one bucket per key.
#[derive(Debug)]
pub struct InMemoryRateLimiter {
    config: RateLimitConfig,
    buckets: Mutex<HashMap<RateKey, Bucket>>,
}

impl InMemoryRateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    fn refill(bucket: &mut Bucket, now: Instant, rate: f64, capacity: f64) {
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * rate).min(capacity);
        bucket.last_refill = now;
    }

    fn snapshot(&self, tokens: f64) -> RateSnapshot {
        let rate = self.config.max_requests as f64 / self.config.window.as_secs_f64();
        let remaining = tokens.floor().max(0.0) as u32;
        let retry_after_seconds = if tokens >= 1.0 {
            0
        } else {
            ((1.0 - tokens) / rate).ceil() as u64
        };
        RateSnapshot {
            limit: self.config.max_requests,
            remaining,
            window_seconds: self.config.window.as_secs(),
            retry_after_seconds,
        }
    }
}

impl RateLimiter for InMemoryRateLimiter {
    fn check(&self, key: &RateKey) -> Result<RateSnapshot, RateLimitError> {
        let capacity = self.config.max_requests as f64;
        let rate = capacity / self.config.window.as_secs_f64();
        let mut buckets = self
            .buckets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        let bucket = buckets.entry(key.clone()).or_insert_with(|| Bucket {
            tokens: capacity,
            last_refill: now,
        });
        Self::refill(bucket, now, rate, capacity);
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            Ok(self.snapshot(bucket.tokens))
        } else {
            let snapshot = self.snapshot(bucket.tokens);
            Err(RateLimitError::Exceeded {
                remaining: snapshot.remaining,
                retry_after_seconds: snapshot.retry_after_seconds,
            })
        }
    }

    fn peek(&self, key: &RateKey) -> RateSnapshot {
        let capacity = self.config.max_requests as f64;
        let rate = capacity / self.config.window.as_secs_f64();
        let mut buckets = self
            .buckets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        let bucket = buckets.entry(key.clone()).or_insert_with(|| Bucket {
            tokens: capacity,
            last_refill: now,
        });
        Self::refill(bucket, now, rate, capacity);
        self.snapshot(bucket.tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(subject: &str) -> RateKey {
        RateKey::new(ScopeId::new("scope"), subject, "chat")
    }

    #[test]
    fn allows_up_to_the_limit_then_refuses() {
        let limiter = InMemoryRateLimiter::new(RateLimitConfig {
            max_requests: 3,
            window: Duration::from_secs(60),
        });
        assert_eq!(limiter.check(&key("a")).unwrap().remaining, 2);
        assert_eq!(limiter.check(&key("a")).unwrap().remaining, 1);
        assert_eq!(limiter.check(&key("a")).unwrap().remaining, 0);
        let error = limiter.check(&key("a")).unwrap_err();
        assert!(matches!(
            error,
            RateLimitError::Exceeded {
                retry_after_seconds,
                ..
            } if retry_after_seconds > 0
        ));
    }

    #[test]
    fn buckets_are_isolated_per_subject_and_namespace() {
        let limiter = InMemoryRateLimiter::new(RateLimitConfig {
            max_requests: 1,
            window: Duration::from_secs(60),
        });
        assert!(limiter.check(&key("a")).is_ok());
        assert!(limiter.check(&key("b")).is_ok());
        let other = RateKey::new(ScopeId::new("scope"), "a", "functions");
        assert!(limiter.check(&other).is_ok());
    }

    #[test]
    fn peek_does_not_consume() {
        let limiter = InMemoryRateLimiter::new(RateLimitConfig {
            max_requests: 2,
            window: Duration::from_secs(60),
        });
        assert_eq!(limiter.peek(&key("a")).remaining, 2);
        assert!(limiter.check(&key("a")).is_ok());
        assert_eq!(limiter.peek(&key("a")).remaining, 1);
    }
}
