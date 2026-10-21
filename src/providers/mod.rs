//! Real provider adapters. Each is behind its own Cargo feature and talks to its vendor's API over HTTPS, so none is compiled by default.
//!
//! - [`anthropic`] (`provider-anthropic`): the Claude Messages API.
//! - [`gemini`] (`provider-gemini`): Google Gemini (`generateContent`, `embedContent`).
//! - [`openai_compat`] (`provider-openai-compat`): any OpenAI-compatible server (Ollama, llama.cpp, a gateway).
//!
//! Every adapter follows the same rules: the credential is a *name* resolved through a [`crate::credentials::CredentialSource`] on each
//! call; errors are mapped to a [`crate::provider::ProviderError`] class and never carry an upstream body or header; a call that would not fit
//! the [`crate::budget::Ledger`] is refused before anything is sent.

use crate::budget::{Ledger, Pricing};
use std::fmt;
use std::sync::Arc;

/// Why an adapter could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// `Locality::Device` but the base URL is not a loopback address.
    DeviceMustBeLoopback,
    /// `Locality::Home` but the host is a public address.
    HomeMustBePrivate,
    /// `Locality::Saas` needs a credential name and a budget.
    SaasNeedsCredentialAndBudget,
    /// A credential name was configured but cannot be loaded now.
    CredentialUnavailable,
    BadUrl,
    HttpClient,
    /// A provider that leaves the home was given prices that are all zero, so the spend ledger could never refuse a call.
    UnpricedProvider,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ConfigError::DeviceMustBeLoopback => "a Device provider must be on a loopback address",
            ConfigError::HomeMustBePrivate => "a Home provider must be on a private address",
            ConfigError::SaasNeedsCredentialAndBudget => {
                "a SaaS provider needs a credential name and a budget"
            }
            ConfigError::CredentialUnavailable => "the configured credential cannot be loaded",
            ConfigError::BadUrl => "the base URL is not valid",
            ConfigError::HttpClient => "the HTTP client could not be built",
            ConfigError::UnpricedProvider => {
                "a remote provider needs declared prices; all-zero pricing cannot be capped"
            }
        })
    }
}
impl std::error::Error for ConfigError {}

/// The spend guard a SaaS adapter must carry.
#[derive(Clone)]
pub struct Budget {
    pub ledger: Arc<Ledger>,
    pub pricing: Pricing,
}

#[cfg(any(feature = "provider-openai-compat", feature = "provider-gemini"))]
impl Budget {
    /// Refuse prices the ledger could not enforce (all zero).
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        if self.pricing.is_unpriced() {
            Err(ConfigError::UnpricedProvider)
        } else {
            Ok(())
        }
    }

    /// The most an embedding call over `inputs` could cost: every input at ~1 token per 3 bytes (rounded up), at the input rate. The same
    /// conservative estimate as text; the true charge comes from the reply's usage when it has one.
    pub(crate) fn embedding_worst_case(&self, inputs: &[String]) -> u64 {
        let tokens: u64 = inputs.iter().map(|t| t.len() as u64 / 3 + 1).sum();
        self.pricing.worst_case(tokens, 0)
    }
}

/// Could the vendor have billed a request that failed with this transport error? A connection that was never made (or a request that
/// could not be built) cannot have been. A timeout, a reset or a reply body that never arrived may have been, so the caller charges the
/// reservation's worst case instead of releasing it.
#[cfg(any(
    feature = "provider-anthropic",
    feature = "provider-openai-compat",
    feature = "provider-gemini"
))]
pub(crate) fn may_have_been_billed(e: &reqwest::Error) -> bool {
    !(e.is_connect() || e.is_builder())
}

#[cfg(feature = "provider-anthropic")]
pub mod anthropic;
#[cfg(feature = "provider-gemini")]
pub mod gemini;
#[cfg(feature = "provider-openai-compat")]
pub mod openai_compat;

#[cfg(test)]
mod conformance_tests;
