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

/// An API-key header value marked sensitive, so `{:?}` of the request's header map prints `Sensitive` instead of the key. (reqwest's
/// `bearer_auth` already does this for `Authorization`; the vendors that use a custom key header need it done by hand.) A key that is
/// not a valid header value is an authentication failure, and the key is not in the error.
#[cfg(any(feature = "provider-anthropic", feature = "provider-gemini"))]
pub(crate) fn sensitive_header_value(
    secret: &str,
) -> Result<reqwest::header::HeaderValue, crate::provider::ProviderError> {
    let mut v = reqwest::header::HeaderValue::from_str(secret)
        .map_err(|_| crate::provider::ProviderError::Auth)?;
    v.set_sensitive(true);
    Ok(v)
}

#[cfg(feature = "provider-anthropic")]
pub mod anthropic;
#[cfg(feature = "provider-gemini")]
pub mod gemini;
#[cfg(feature = "provider-openai-compat")]
pub mod openai_compat;

#[cfg(test)]
mod conformance_tests;

#[cfg(all(test, any(feature = "provider-anthropic", feature = "provider-gemini")))]
mod header_tests {
    use super::sensitive_header_value;

    #[test]
    fn an_api_key_header_is_sensitive_and_does_not_show_in_debug_output() {
        const KEY: &str = "sk-TEST-ONLY-header-key";
        let req = reqwest::Client::new()
            .get("http://127.0.0.1:9/")
            .header("x-api-key", sensitive_header_value(KEY).unwrap())
            .build()
            .unwrap();
        let shown = format!("{:?} {:?}", req.headers(), req);
        assert!(!shown.contains(KEY), "{shown}");
        assert!(shown.contains("Sensitive"), "{shown}");
        // it is still the real value on the wire
        let v = req.headers().get("x-api-key").unwrap();
        assert!(v.is_sensitive());
        assert_eq!(v.to_str().unwrap(), KEY);
    }

    #[test]
    fn a_key_that_is_not_a_valid_header_value_is_an_auth_error_without_the_key() {
        let e = sensitive_header_value("bad\nkey").unwrap_err();
        assert_eq!(e, crate::provider::ProviderError::Auth);
    }
}
