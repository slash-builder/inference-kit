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
    /// A provider that leaves the home was given prices that cannot cap every kind of request it offers (a zero input or output rate), so
    /// the spend ledger could never refuse such a call.
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
                "a remote provider needs a declared price for every kind of request it offers; a zero rate cannot be capped"
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

/// Refuse prices the ledger could not enforce for the request `kinds` an adapter offers: a kind whose reservation would come to zero
/// (a missing input or output rate) is never capped, so the adapter must not be built. A rate of zero is a declaration of "free", and a
/// provider that leaves the home is never taken at its word on that.
#[cfg(any(
    feature = "provider-anthropic",
    feature = "provider-openai-compat",
    feature = "provider-gemini"
))]
pub(crate) fn validate_pricing(
    pricing: &Pricing,
    kinds: &[crate::modality::RequestKind],
) -> Result<(), ConfigError> {
    if kinds.iter().all(|k| pricing.can_cap(*k)) {
        Ok(())
    } else {
        Err(ConfigError::UnpricedProvider)
    }
}

/// How long a connection may take to be made before the request is given up as never sent.
#[cfg(any(
    feature = "provider-anthropic",
    feature = "provider-openai-compat",
    feature = "provider-gemini"
))]
pub(crate) const DEFAULT_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// The HTTP client every adapter starts from. The connect phase has its own, shorter limit so that a host that never answers is
/// reported as a connection that was never made (the request was not sent, so it is not billed), not as a timeout of a request that may
/// have been.
#[cfg(any(
    feature = "provider-anthropic",
    feature = "provider-openai-compat",
    feature = "provider-gemini"
))]
pub(crate) fn http_builder(
    timeout: std::time::Duration,
    connect_timeout: std::time::Duration,
) -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(connect_timeout)
        .redirect(reqwest::redirect::Policy::none())
}

#[cfg(any(feature = "provider-openai-compat", feature = "provider-gemini"))]
impl Budget {
    /// Refuse prices the ledger could not enforce for these request kinds (see [`validate_pricing`]).
    pub(crate) fn validate(
        &self,
        kinds: &[crate::modality::RequestKind],
    ) -> Result<(), ConfigError> {
        validate_pricing(&self.pricing, kinds)
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

/// Helpers for the adapters' tests.
#[cfg(all(
    test,
    any(
        feature = "provider-anthropic",
        feature = "provider-openai-compat",
        feature = "provider-gemini"
    )
))]
pub(crate) mod test_support {
    use std::net::{SocketAddr, TcpStream};
    use std::time::Duration;

    /// The `host:port` of an address on which a connection attempt hangs (packets are dropped, nothing refuses), or `None` where that is
    /// not so (no route, a refusal, or something answers). It is probed, not assumed: a test that needs a connect timeout must skip, not
    /// fail, on a machine whose network does not blackhole this address.
    pub(crate) fn blackholed_addr() -> Option<String> {
        let addr: SocketAddr = "10.255.255.1:81".parse().ok()?;
        match TcpStream::connect_timeout(&addr, Duration::from_millis(500)) {
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => Some(addr.to_string()),
            _ => {
                eprintln!("skipped: {addr} is not blackholed on this network");
                None
            }
        }
    }
}

/// The key header value, marked sensitive (see `crate::sensitive_header`). A key that is not a valid header value is an
/// authentication failure, and the key is not in the error.
#[cfg(any(feature = "provider-anthropic", feature = "provider-gemini"))]
pub(crate) fn sensitive_header_value(
    secret: &str,
) -> Result<reqwest::header::HeaderValue, crate::provider::ProviderError> {
    crate::sensitive_header::sensitive_header_value(secret)
        .ok_or(crate::provider::ProviderError::Auth)
}

#[cfg(feature = "provider-anthropic")]
pub mod anthropic;
#[cfg(feature = "provider-gemini")]
pub mod gemini;
#[cfg(feature = "provider-openai-compat")]
pub mod openai_compat;

#[cfg(test)]
mod conformance_tests;
