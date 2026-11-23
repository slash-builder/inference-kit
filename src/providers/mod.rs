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
/// have been. That only works if the connect limit ends first: when `connect_timeout >= timeout` the overall timer would end a hung
/// connect, as a timeout that is not a connect error, and the call would be charged for a request that was never sent. So the connect
/// limit is **clamped to half the overall timeout** in that case ([`effective_connect_timeout`]).
///
/// **Redirects are never followed.** A redirect from the endpoint a provider was configured for would send the request, prompt included
/// (a 307 or 308 replays the body), to an origin nobody configured, checked against the egress policy or audited. An API endpoint that
/// redirects is a misconfiguration, and it surfaces as an error.
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
        .connect_timeout(effective_connect_timeout(timeout, connect_timeout))
        .redirect(reqwest::redirect::Policy::none())
}

/// The connect limit a client is built with: the configured one, unless it is not shorter than the overall timeout, in which case half
/// the overall timeout (so the connect limit, not the overall timer, ends a connection that hangs).
#[cfg(any(
    feature = "provider-anthropic",
    feature = "provider-openai-compat",
    feature = "provider-gemini"
))]
pub(crate) fn effective_connect_timeout(
    timeout: std::time::Duration,
    connect_timeout: std::time::Duration,
) -> std::time::Duration {
    if connect_timeout < timeout {
        connect_timeout
    } else {
        timeout / 2
    }
}

/// Could the vendor have billed a request that came back with this HTTP status? A request refused for its content, its credentials or its
/// rate (4xx), or turned away because the service is overloaded or unavailable (500, 503, 529), was not processed to completion and is not
/// billed. **408, 502 and 504 are different**: they are what a proxy or gateway in front of the vendor says when it gave up waiting, and the
/// vendor may have finished and billed the call anyway, so they are charged at the reserved worst case like any other outcome that is
/// unknown. Any other status is released.
#[cfg(any(
    feature = "provider-anthropic",
    feature = "provider-openai-compat",
    feature = "provider-gemini"
))]
pub(crate) fn status_may_have_been_billed(status: u16) -> bool {
    matches!(status, 408 | 502 | 504)
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
    use wiremock::matchers::any;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Two local servers: the first answers every request with `status` and a `Location` pointing at the second, which would answer 200.
    /// A client that follows the redirect delivers a request to the second; one that does not leaves it untouched.
    pub(crate) async fn redirector_and_trap(status: u16) -> (MockServer, MockServer) {
        let trap = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&trap)
            .await;
        let redirector = MockServer::start().await;
        Mock::given(any())
            .respond_with(
                ResponseTemplate::new(status)
                    .insert_header("location", format!("{}/v1/redirected", trap.uri())),
            )
            .mount(&redirector)
            .await;
        (redirector, trap)
    }

    /// How many requests reached the server.
    pub(crate) async fn received(s: &MockServer) -> usize {
        s.received_requests().await.map_or(0, |r| r.len())
    }

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

    /// A server that sends the status line and headers of a 200 reply promising a long body, sends one byte of it and closes. The request
    /// was accepted and read, then the reply broke: reading the body fails. Returns `host:port`.
    pub(crate) fn truncated_reply_server() -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a loopback port");
        let addr = listener.local_addr().expect("local address").to_string();
        std::thread::spawn(move || {
            while let Ok((mut conn, _)) = listener.accept() {
                let mut buf = [0u8; 65536];
                // enough of the request to have been "received" (the small test bodies arrive with the headers)
                let _ = conn.read(&mut buf);
                let _ = conn.write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 1000\r\n\r\n{",
                );
                let _ = conn.flush();
                // closing here ends the body early
            }
        });
        addr
    }

    /// A credential source that works until told to fail: the adapter is built with it, then the credential disappears before the call.
    pub(crate) struct Vanishing {
        pub(crate) value: String,
        pub(crate) gone: std::sync::atomic::AtomicBool,
    }
    impl Vanishing {
        pub(crate) fn new(value: &str) -> std::sync::Arc<Self> {
            std::sync::Arc::new(Self {
                value: value.into(),
                gone: std::sync::atomic::AtomicBool::new(false),
            })
        }
        pub(crate) fn vanish(&self) {
            self.gone.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    impl crate::credentials::CredentialSource for Vanishing {
        fn get(
            &self,
            name: &str,
        ) -> Result<crate::credentials::Secret, crate::credentials::CredentialError> {
            if self.gone.load(std::sync::atomic::Ordering::SeqCst) {
                Err(crate::credentials::CredentialError::Unconfigured(
                    name.into(),
                ))
            } else {
                Ok(crate::credentials::Secret::new(self.value.clone()))
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

#[cfg(all(
    test,
    any(
        feature = "provider-anthropic",
        feature = "provider-openai-compat",
        feature = "provider-gemini"
    )
))]
mod tests {
    use super::test_support::{blackholed_addr, truncated_reply_server};
    use super::*;
    use std::time::Duration;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn client(timeout: Duration, connect: Duration) -> reqwest::Client {
        http_builder(timeout, connect).no_proxy().build().unwrap()
    }

    #[tokio::test]
    async fn a_request_that_could_not_be_built_was_never_sent() {
        let e = client(Duration::from_secs(5), Duration::from_secs(1))
            .get("http://exa mple/")
            .send()
            .await
            .unwrap_err();
        assert!(e.is_builder() && !e.is_connect(), "premise: {e:?}");
        assert!(
            !may_have_been_billed(&e),
            "a request that was not built was not sent"
        );
    }

    #[tokio::test]
    async fn a_refused_connection_was_never_made() {
        let e = client(Duration::from_secs(5), Duration::from_secs(1))
            .get("http://127.0.0.1:1/")
            .send()
            .await
            .unwrap_err();
        assert!(e.is_connect() && !e.is_builder(), "premise: {e:?}");
        assert!(!may_have_been_billed(&e));
    }

    #[tokio::test]
    async fn a_timeout_after_the_request_was_sent_and_a_broken_reply_may_have_been_billed() {
        let s = MockServer::start().await;
        Mock::given(wiremock::matchers::any())
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
            .mount(&s)
            .await;
        let e = client(Duration::from_millis(300), Duration::from_millis(100))
            .get(s.uri())
            .send()
            .await
            .unwrap_err();
        assert!(e.is_timeout() && !e.is_connect(), "premise: {e:?}");
        assert!(may_have_been_billed(&e));

        let resp = client(Duration::from_secs(5), Duration::from_secs(1))
            .get(format!("http://{}/", truncated_reply_server()))
            .send()
            .await
            .unwrap();
        let e = resp.text().await.unwrap_err();
        assert!(!e.is_connect() && !e.is_builder(), "premise: {e:?}");
        assert!(may_have_been_billed(&e));
    }

    #[test]
    fn the_connect_limit_is_always_shorter_than_the_overall_one() {
        let ms = Duration::from_millis;
        assert_eq!(effective_connect_timeout(ms(1000), ms(300)), ms(300));
        // not shorter: clamped to half, so the connect limit still ends a hung connection first
        assert_eq!(effective_connect_timeout(ms(300), ms(5000)), ms(150));
        assert_eq!(effective_connect_timeout(ms(300), ms(300)), ms(150));
    }

    #[tokio::test]
    async fn a_hung_connection_is_a_connect_error_even_when_the_connect_limit_was_set_too_long() {
        let Some(addr) = blackholed_addr() else {
            return;
        };
        // connect limit (5 s) longer than the overall timeout (400 ms): without the clamp the overall timer ends it
        let e = client(Duration::from_millis(400), Duration::from_secs(5))
            .get(format!("http://{addr}/"))
            .send()
            .await
            .unwrap_err();
        assert!(e.is_timeout() && e.is_connect(), "{e:?}");
        assert!(!may_have_been_billed(&e));
    }

    #[test]
    fn the_default_connect_timeout_is_ten_seconds() {
        assert_eq!(DEFAULT_CONNECT_TIMEOUT, Duration::from_secs(10));
    }

    #[test]
    fn only_the_statuses_a_gateway_uses_for_giving_up_may_have_been_billed() {
        for s in [408u16, 502, 504] {
            assert!(status_may_have_been_billed(s), "{s}");
        }
        for s in [
            200u16, 301, 400, 401, 403, 404, 413, 422, 429, 500, 501, 503, 529,
        ] {
            assert!(!status_may_have_been_billed(s), "{s}");
        }
    }

    #[test]
    fn validate_pricing_checks_every_kind_offered() {
        use crate::modality::RequestKind::{Embedding, TextGeneration};
        let p = Pricing {
            input_micro_per_mtok: 1,
            output_micro_per_mtok: 0,
            ..Pricing::default()
        };
        assert_eq!(
            validate_pricing(&p, &[TextGeneration]),
            Err(ConfigError::UnpricedProvider)
        );
        assert_eq!(validate_pricing(&p, &[Embedding]), Ok(()));
    }
}
