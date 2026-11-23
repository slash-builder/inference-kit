//! An adapter for any OpenAI-compatible server: Ollama, the llama.cpp server, a gateway. **Chat completions and embeddings.**
//!
//! One adapter, three very different places it can point, and the difference is **where the content goes**, so it is declared, not guessed:
//!
//! - [`Locality::Device`]: this machine. The base URL **must be a loopback address**; anything else is refused when the adapter is built
//!   (a server reachable from the network is not "this device").
//! - [`Locality::Home`]: another machine in the household. Allowed on a private (LAN) address; the egress policy still sees it as staying
//!   in the home.
//! - [`Locality::Saas`]: a third party. Requires a credential **name** and a spend budget ([`Ledger`] + [`Pricing`]); the content leaves
//!   the home, so the egress policy must allow it hop by hop.
//!
//! A credential, when there is one, is read through the [`CredentialSource`] on every call and sent as a bearer token; a local server needs
//! none. Errors are classes only, never the upstream body. An OpenAI-style `message.refusal` is a *result* ([`FinishReason::Refusal`]),
//! and a `content_filter` finish ends the chain too.
//!
//! Not in this slice: streaming, tool calls, images, structured output, capacity probing (`/api/ps`). Tested against wiremock; one
//! `#[ignore]` test runs against a real local Ollama.

use crate::budget::Usage;
use crate::credentials::CredentialSource;
use crate::egress::{Locality, ProviderInfo};
use crate::modality::{EmbeddingResponse, Output, Request, RequestKind};
use crate::model_card::{Cap, CardSource, ModelCard};
use crate::provider::{FinishReason, ProviderAdapter, ProviderError, ProviderReply};
use crate::types::InferenceResponse;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::fmt;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

pub use super::{Budget, ConfigError};

#[derive(Debug, Clone)]
pub struct OpenAiCompatConfig {
    pub provider_id: String,
    /// For example `http://127.0.0.1:11434` (Ollama). The adapter appends `/v1/...`.
    pub base_url: String,
    pub locality: Locality,
    /// The model for chat requests.
    pub model: String,
    /// The model for embedding requests; if `None` the adapter does not offer embeddings.
    pub embedding_model: Option<String>,
    /// Credential NAME, if the server needs a bearer token (a local server usually does not).
    pub credential_name: Option<String>,
    pub default_max_tokens: u32,
    /// The limit on the whole request, reply included. For a remote provider, a timeout after the request was sent is charged at the
    /// reserved worst case.
    pub timeout: Duration,
    /// The limit on making the connection. A host that does not answer within it is a request that was never sent: released, not charged.
    pub connect_timeout: Duration,
}

impl OpenAiCompatConfig {
    /// Defaults for a local Ollama.
    pub fn ollama(model: &str) -> Self {
        Self {
            provider_id: "ollama".into(),
            base_url: "http://127.0.0.1:11434".into(),
            locality: Locality::Device,
            model: model.into(),
            embedding_model: None,
            credential_name: None,
            default_max_tokens: 1024,
            timeout: Duration::from_secs(300),
            connect_timeout: super::DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

pub struct OpenAiCompatAdapter {
    info: ProviderInfo,
    cfg: OpenAiCompatConfig,
    creds: Option<Arc<dyn CredentialSource>>,
    budget: Option<Budget>,
    http: reqwest::Client,
}

impl fmt::Debug for OpenAiCompatAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiCompatAdapter")
            .field("provider_id", &self.cfg.provider_id)
            .field("locality", &self.cfg.locality)
            .field("model", &self.cfg.model)
            .finish()
    }
}

/// Why a POST failed, and whether the vendor may nevertheless have billed it.
struct PostFailure {
    err: ProviderError,
    maybe_billed: bool,
}

fn host_of(url: &str) -> Option<String> {
    let rest = url.split("://").nth(1)?;
    let authority = rest.split('/').next()?;
    let authority = authority.rsplit('@').next()?;
    // [::1]:11434, 127.0.0.1:11434, localhost:11434, host
    let host = if let Some(stripped) = authority.strip_prefix('[') {
        stripped.split(']').next()?
    } else {
        authority.split(':').next()?
    };
    (!host.is_empty()).then(|| host.to_string())
}

fn is_loopback(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn is_private(host: &str) -> bool {
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(v4)) => v4.is_private() || v4.is_loopback() || v4.is_link_local(),
        Ok(IpAddr::V6(v6)) => v6.is_loopback() || (v6.segments()[0] & 0xfe00) == 0xfc00,
        // a name: only the usual home-network suffixes count as the household
        Err(_) => {
            host.eq_ignore_ascii_case("localhost")
                || host.ends_with(".local")
                || host.ends_with(".lan")
                || host.ends_with(".home.arpa")
        }
    }
}

impl OpenAiCompatAdapter {
    /// Build the adapter. **Fails closed** on a locality that does not match the address, a SaaS (or unspecified) provider without a budget,
    /// a provider that leaves the home whose prices cannot cap every kind of request it offers (text needs an input and an output rate;
    /// embeddings, when configured, need the input rate), and a credential that cannot be loaded.
    pub fn new(
        cfg: OpenAiCompatConfig,
        creds: Option<Arc<dyn CredentialSource>>,
        budget: Option<Budget>,
    ) -> Result<Self, ConfigError> {
        Self::build(cfg, creds, budget, |b| b)
    }

    /// `new`, with a hook on the HTTP client builder (tests use it to keep a proxy in the environment out of the way).
    fn build(
        cfg: OpenAiCompatConfig,
        creds: Option<Arc<dyn CredentialSource>>,
        budget: Option<Budget>,
        tweak: impl FnOnce(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
    ) -> Result<Self, ConfigError> {
        let mut kinds = vec![RequestKind::TextGeneration];
        if cfg.embedding_model.is_some() {
            kinds.push(RequestKind::Embedding);
        }
        if let Some(b) = &budget {
            if cfg.locality.leaves_home() {
                b.validate(&kinds)?;
            }
        }
        let host = host_of(&cfg.base_url).ok_or(ConfigError::BadUrl)?;
        match cfg.locality {
            Locality::Device if !is_loopback(&host) => {
                return Err(ConfigError::DeviceMustBeLoopback)
            }
            Locality::Home if !is_private(&host) => return Err(ConfigError::HomeMustBePrivate),
            Locality::Saas
                if cfg.credential_name.is_none() || budget.is_none() || creds.is_none() =>
            {
                return Err(ConfigError::SaasNeedsCredentialAndBudget)
            }
            // an unspecified locality is the strictest case (remote), so it carries the same requirements
            Locality::Unspecified
                if cfg.credential_name.is_none() || budget.is_none() || creds.is_none() =>
            {
                return Err(ConfigError::SaasNeedsCredentialAndBudget)
            }
            _ => {}
        }
        if let (Some(name), Some(c)) = (&cfg.credential_name, &creds) {
            c.get(name)
                .map_err(|_| ConfigError::CredentialUnavailable)?;
        }
        let http = tweak(super::http_builder(cfg.timeout, cfg.connect_timeout))
            .build()
            .map_err(|_| ConfigError::HttpClient)?;
        Ok(Self {
            info: ProviderInfo {
                provider_id: cfg.provider_id.clone(),
                locality: cfg.locality,
                kinds,
            },
            cfg,
            creds,
            budget,
            http,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.cfg.base_url.trim_end_matches('/'), path)
    }

    fn auth(&self) -> Result<Option<String>, ProviderError> {
        match (&self.cfg.credential_name, &self.creds) {
            (Some(name), Some(c)) => c
                .get(name)
                .map(|s| Some(s.expose().to_string()))
                .map_err(|_| ProviderError::Auth),
            _ => Ok(None),
        }
    }

    fn transport(&self, e: reqwest::Error) -> ProviderError {
        if e.is_timeout() {
            ProviderError::Timeout
        } else if e.is_connect() && !self.cfg.locality.leaves_home() {
            // a local or household server that is not listening is "not available", not a network fault
            ProviderError::Unavailable
        } else {
            ProviderError::Network
        }
    }

    /// `GET /v1/models` as cards. Capacity facts are not in this endpoint, so they stay unknown. `today` is `YYYY-MM-DD`.
    pub async fn discover_models(&self, today: &str) -> Result<Vec<ModelCard>, ProviderError> {
        let mut rb = self.http.get(self.url("/v1/models"));
        if let Some(k) = self.auth()? {
            rb = rb.bearer_auth(k);
        }
        let resp = rb.send().await.map_err(|e| self.transport(e))?;
        let status = resp.status().as_u16();
        let retry = retry_after(resp.headers());
        let body: Value = serde_json::from_str(&resp.text().await.map_err(|e| self.transport(e))?)
            .unwrap_or(Value::Null);
        if status != 200 {
            return Err(classify(status, retry, &body));
        }
        Ok(body["data"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|m| m["id"].as_str())
                    .map(|id| ModelCard {
                        provider_id: self.cfg.provider_id.clone(),
                        model_id: id.to_string(),
                        kinds: self.info.kinds.clone(),
                        max_input_tokens: None,
                        max_output_tokens: None,
                        retires_at: None,
                        image_input: Cap::Unknown,
                        structured_outputs: Cap::Unknown,
                        tool_use: Cap::Unknown,
                        source: CardSource::Probed {
                            on: today.to_string(),
                        },
                    })
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Is the server answering? (`GET /v1/models`)
    pub async fn health(&self) -> bool {
        self.discover_models("").await.is_ok()
    }

    /// POST and parse. On failure the error says whether the request may have reached the vendor (`maybe_billed`): a connection that was
    /// never made, or a refusal to even build the request, cannot have been.
    async fn post(&self, path: &str, body: &Value) -> Result<(u64, Value, u64), PostFailure> {
        let not_sent = |err| PostFailure {
            err,
            maybe_billed: false,
        };
        let bytes = serde_json::to_vec(body).map_err(|_| not_sent(ProviderError::Other))?;
        let sent = bytes.len() as u64;
        let mut rb = self
            .http
            .post(self.url(path))
            .header("content-type", "application/json")
            .body(bytes);
        if let Some(k) = self.auth().map_err(not_sent)? {
            rb = rb.bearer_auth(k);
        }
        let transport_failure = |e: reqwest::Error| PostFailure {
            maybe_billed: super::may_have_been_billed(&e),
            err: self.transport(e),
        };
        let resp = rb.send().await.map_err(transport_failure)?;
        let status = resp.status().as_u16();
        let retry = retry_after(resp.headers());
        let text = resp.text().await.map_err(transport_failure)?;
        let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if status != 200 {
            // an error status is not billed, except the few a gateway uses when it gave up on a call the vendor may have finished
            return Err(PostFailure {
                err: classify(status, retry, &parsed),
                maybe_billed: super::status_may_have_been_billed(status),
            });
        }
        Ok((sent, parsed, text.len() as u64))
    }

    async fn chat(
        &self,
        p: &crate::types::InferenceParams,
    ) -> Result<ProviderReply, ProviderError> {
        let max_tokens = if p.max_tokens == 0 {
            self.cfg.default_max_tokens
        } else {
            p.max_tokens
        };
        let reservation = match &self.budget {
            Some(b) => Some(
                b.ledger.reserve(
                    b.pricing
                        .worst_case(p.prompt.len() as u64 / 3 + 1, u64::from(max_tokens)),
                )?,
            ),
            None => None,
        };
        let body = json!({
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": p.prompt}],
            "max_tokens": max_tokens,
            "temperature": p.temperature,
            "stream": false,
        });
        let (sent, resp, received) = match self.post("/v1/chat/completions", &body).await {
            Ok(r) => r,
            Err(f) => {
                if let (Some(b), Some(r)) = (&self.budget, reservation) {
                    if f.maybe_billed {
                        b.ledger.settle_reserved(r);
                    } else {
                        b.ledger.release(r);
                    }
                }
                return Err(f.err);
            }
        };
        let usage = Usage {
            input_tokens: resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: resp["usage"]["completion_tokens"].as_u64().unwrap_or(0),
            ..Usage::default()
        };
        if let (Some(b), Some(r)) = (&self.budget, reservation) {
            if resp["usage"]["prompt_tokens"].is_u64()
                && resp["usage"]["completion_tokens"].is_u64()
            {
                b.ledger.settle(r, b.pricing.cost(&usage));
            } else {
                // no usage in a billed reply: charge the worst case, never zero
                b.ledger.settle_reserved(r);
            }
        }
        let choice = &resp["choices"][0];
        let message = &choice["message"];
        let refused = message["refusal"].as_str().is_some_and(|s| !s.is_empty());
        let finish = if refused {
            FinishReason::Refusal
        } else {
            match choice["finish_reason"].as_str().unwrap_or("stop") {
                "length" => FinishReason::Length,
                "tool_calls" | "function_call" => FinishReason::ToolCalls,
                "content_filter" => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
            }
        };
        let text = if refused {
            message["refusal"].as_str().unwrap_or("")
        } else {
            message["content"].as_str().unwrap_or("")
        };
        let served = resp["model"]
            .as_str()
            .unwrap_or(&self.cfg.model)
            .to_string();
        Ok(ProviderReply {
            output: Output::Text(InferenceResponse {
                completion: text.to_string(),
                total_tokens: (usage.input_tokens + usage.output_tokens) as u32,
                prompt_tokens: usage.input_tokens as u32,
                completion_tokens: usage.output_tokens as u32,
                inference_ms: 0,
                model_version: served.clone(),
                quantization: "n/a (server)".into(),
                session_id: Default::default(),
                request_id: String::new(),
            }),
            finish,
            model_served: served,
            bytes_sent: sent,
            bytes_received: received,
        })
    }

    async fn embed(
        &self,
        p: &crate::modality::EmbeddingParams,
    ) -> Result<ProviderReply, ProviderError> {
        let model = self
            .cfg
            .embedding_model
            .as_deref()
            .ok_or(ProviderError::Other)?;
        // a remote embedding call is paid for like a chat call: reserve first, refuse before sending if it does not fit
        let reservation = match &self.budget {
            Some(b) => Some(b.ledger.reserve(b.embedding_worst_case(&p.inputs))?),
            None => None,
        };
        let posted = self
            .post(
                "/v1/embeddings",
                &json!({"model": model, "input": p.inputs}),
            )
            .await;
        let (sent, resp, received) = match posted {
            Ok(r) => r,
            Err(f) => {
                if let (Some(b), Some(r)) = (&self.budget, reservation) {
                    if f.maybe_billed {
                        b.ledger.settle_reserved(r);
                    } else {
                        b.ledger.release(r);
                    }
                }
                return Err(f.err);
            }
        };
        // settle from the reply's usage when it has one, else at the reserved estimate
        if let (Some(b), Some(r)) = (&self.budget, reservation) {
            match resp["usage"]["prompt_tokens"].as_u64() {
                Some(t) => b.ledger.settle(
                    r,
                    b.pricing.cost(&Usage {
                        input_tokens: t,
                        ..Usage::default()
                    }),
                ),
                None => b.ledger.settle_reserved(r),
            }
        }
        let mut vectors: Vec<Vec<f32>> = resp["data"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|d| {
                        d["embedding"]
                            .as_array()
                            .map(|v| {
                                v.iter()
                                    .filter_map(Value::as_f64)
                                    .map(|x| x as f32)
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default();
        if vectors.len() != p.inputs.len() || vectors.iter().any(Vec::is_empty) {
            return Err(ProviderError::Other);
        }
        if p.normalize {
            for v in &mut vectors {
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if n > 0.0 {
                    v.iter_mut().for_each(|x| *x /= n);
                }
            }
        }
        let served = resp["model"].as_str().unwrap_or(model).to_string();
        let dimensions = vectors.first().map_or(0, Vec::len);
        Ok(ProviderReply {
            output: Output::Embeddings(EmbeddingResponse {
                vectors,
                dimensions,
                prompt_tokens: resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                inference_ms: 0,
                model_version: served.clone(),
                session_id: Default::default(),
                request_id: String::new(),
            }),
            finish: FinishReason::Stop,
            model_served: served,
            bytes_sent: sent,
            bytes_received: received,
        })
    }
}

fn retry_after(h: &reqwest::header::HeaderMap) -> Option<u32> {
    h.get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

/// Map an error response to a class. The body is read only to tell a missing model and a context overflow apart; it is never kept.
pub(crate) fn classify(status: u16, retry_after: Option<u32>, body: &Value) -> ProviderError {
    let message = body["error"]["message"]
        .as_str()
        .or_else(|| body["error"].as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match status {
        400 if message.contains("context")
            || message.contains("too long")
            || message.contains("maximum") =>
        {
            ProviderError::ContextOverflow
        }
        401 | 403 => ProviderError::Auth,
        402 => ProviderError::BudgetExceeded,
        404 if message.contains("model") => ProviderError::ModelNotFound,
        413 => ProviderError::ContextOverflow,
        429 => ProviderError::RateLimited {
            retry_after_s: retry_after,
        },
        503 | 529 => ProviderError::Overloaded,
        504 => ProviderError::Timeout,
        500..=599 => ProviderError::Unavailable,
        _ => ProviderError::Other,
    }
}

#[async_trait]
impl ProviderAdapter for OpenAiCompatAdapter {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }
    fn model_id(&self) -> &str {
        &self.cfg.model
    }
    async fn call(&self, req: &Request) -> Result<ProviderReply, ProviderError> {
        match req {
            Request::Text(p) => self.chat(p).await,
            Request::Embedding(p) => self.embed(p).await,
            _ => Err(ProviderError::Other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{Ledger, Pricing};
    use crate::credentials::InMemoryCredentials;
    use crate::egress::{AgeBand, Label, Policy, Purpose, SourceKind, Subject, VecSink};
    use crate::modality::EmbeddingParams;
    use crate::model_card::ModelCard;
    use crate::provider::{run_chain, ChainConfig};
    use crate::types::InferenceParams;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg(server: &MockServer) -> OpenAiCompatConfig {
        let mut c = OpenAiCompatConfig::ollama("test-model");
        c.base_url = server.uri(); // wiremock binds 127.0.0.1
        c
    }
    fn text(prompt: &str) -> Request {
        Request::Text(InferenceParams {
            model_id: "m".into(),
            prompt: prompt.into(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 50,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        })
    }
    fn chat_ok(content: &str, finish: &str) -> Value {
        json!({"id":"c1","model":"test-model:served","choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":finish}],
               "usage":{"prompt_tokens":7,"completion_tokens":3}})
    }
    async fn serve(s: &MockServer, status: u16, body: Value) {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(s)
            .await;
    }
    fn out(r: &ProviderReply) -> String {
        match &r.output {
            Output::Text(t) => t.completion.clone(),
            o => panic!("{o:?}"),
        }
    }

    #[tokio::test]
    async fn a_local_chat_answer_maps_text_usage_and_what_was_served() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(
                json!({"model":"test-model","stream":false,"max_tokens":50}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_ok("hello there", "stop")))
            .expect(1)
            .mount(&s)
            .await;
        let a = OpenAiCompatAdapter::new(cfg(&s), None, None).unwrap();
        let r = a.call(&text("hi")).await.unwrap();
        assert_eq!(
            (out(&r), r.finish),
            ("hello there".to_string(), FinishReason::Stop)
        );
        assert_eq!(r.model_served, "test-model:served");
        assert!(r.bytes_sent > 0 && r.bytes_received > 0);
        match r.output {
            Output::Text(t) => assert_eq!((t.prompt_tokens, t.completion_tokens), (7, 3)),
            _ => unreachable!(),
        }
        // a local server needs no credential: no Authorization header is sent
        let reqs = s.received_requests().await.unwrap();
        assert!(reqs[0].headers.get("authorization").is_none());
    }

    #[tokio::test]
    async fn finish_reasons_refusals_and_filters_are_results() {
        for (finish, want) in [
            ("length", FinishReason::Length),
            ("tool_calls", FinishReason::ToolCalls),
            ("content_filter", FinishReason::ContentFilter),
            ("stop", FinishReason::Stop),
        ] {
            let s = MockServer::start().await;
            serve(&s, 200, chat_ok("x", finish)).await;
            let a = OpenAiCompatAdapter::new(cfg(&s), None, None).unwrap();
            assert_eq!(a.call(&text("q")).await.unwrap().finish, want, "{finish}");
        }
        let s = MockServer::start().await;
        serve(&s, 200, json!({"model":"m","choices":[{"message":{"role":"assistant","content":null,"refusal":"I can't do that"},"finish_reason":"stop"}],"usage":{}})).await;
        let a = OpenAiCompatAdapter::new(cfg(&s), None, None).unwrap();
        let r = a.call(&text("q")).await.unwrap();
        assert_eq!(
            (r.finish, out(&r)),
            (FinishReason::Refusal, "I can't do that".to_string())
        );
        assert!(r.finish != FinishReason::Stop);
    }

    #[tokio::test]
    async fn errors_map_to_classes_and_never_carry_the_body() {
        let cases = [
            (
                404,
                json!({"error":{"message":"model 'nope' not found, try pulling it first"}}),
                ProviderError::ModelNotFound,
            ),
            (
                401,
                json!({"error":{"message":"bad key SECRET-UPSTREAM-TEXT"}}),
                ProviderError::Auth,
            ),
            (
                429,
                json!({"error":{"message":"slow"}}),
                ProviderError::RateLimited {
                    retry_after_s: None,
                },
            ),
            (
                400,
                json!({"error":{"message":"request exceeds the maximum context length"}}),
                ProviderError::ContextOverflow,
            ),
            (500, json!({"error":"boom"}), ProviderError::Unavailable),
            (503, json!({}), ProviderError::Overloaded),
            (418, json!({}), ProviderError::Other),
        ];
        for (status, body, want) in cases {
            let s = MockServer::start().await;
            serve(&s, status, body).await;
            let a = OpenAiCompatAdapter::new(cfg(&s), None, None).unwrap();
            assert_eq!(a.call(&text("q")).await.unwrap_err(), want, "HTTP {status}");
        }
        // through the executor: the upstream text never reaches the caller
        let s = MockServer::start().await;
        serve(&s, 401, json!({"error":{"message":"SECRET-UPSTREAM-TEXT"}})).await;
        let a: Arc<dyn ProviderAdapter> =
            Arc::new(OpenAiCompatAdapter::new(cfg(&s), None, None).unwrap());
        let label = Label::new(
            SourceKind::HouseholdFile,
            &["notes"],
            &[Subject::person("a", AgeBand::Adult)],
        );
        let e = run_chain(
            &[a],
            &Policy::default(),
            &label,
            &text("q"),
            &ChainConfig::new(Purpose::Answer, 1),
            &VecSink::default(),
        )
        .await
        .unwrap_err();
        assert!(!format!("{e} {e:?}").contains("SECRET"));
    }

    #[tokio::test]
    async fn a_local_provider_needs_no_policy_rule_but_a_stopped_server_is_unavailable_not_a_network_fault(
    ) {
        // a closed port on loopback: connection refused
        let mut c = OpenAiCompatConfig::ollama("m");
        c.base_url = "http://127.0.0.1:9".into();
        c.timeout = Duration::from_secs(3);
        let a = OpenAiCompatAdapter::new(c, None, None).unwrap();
        assert_eq!(
            a.call(&text("q")).await.unwrap_err(),
            ProviderError::Unavailable
        );
        assert!(!a.health().await);
        // the egress policy lets a Device provider run with no rules at all
        let s = MockServer::start().await;
        serve(&s, 200, chat_ok("ok", "stop")).await;
        let live: Arc<dyn ProviderAdapter> =
            Arc::new(OpenAiCompatAdapter::new(cfg(&s), None, None).unwrap());
        let label = Label::new(
            SourceKind::HouseholdFile,
            &["health"],
            &[Subject::person("kid", AgeBand::Minor)],
        );
        let (reply, _) = run_chain(
            &[live],
            &Policy::default(),
            &label,
            &text("q"),
            &ChainConfig::new(Purpose::Answer, 1),
            &VecSink::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            out(&reply),
            "ok",
            "private content may be answered by a provider on this device"
        );
    }

    #[test]
    fn the_locality_must_match_the_address() {
        let mk = |url: &str, loc: Locality| {
            let mut c = OpenAiCompatConfig::ollama("m");
            c.base_url = url.into();
            c.locality = loc;
            OpenAiCompatAdapter::new(c, None, None).err()
        };
        assert_eq!(
            mk("http://192.168.1.20:11434", Locality::Device),
            Some(ConfigError::DeviceMustBeLoopback)
        );
        assert_eq!(
            mk("http://example.com", Locality::Device),
            Some(ConfigError::DeviceMustBeLoopback)
        );
        assert!(mk("http://localhost:11434", Locality::Device).is_none());
        assert!(mk("http://[::1]:11434", Locality::Device).is_none());
        assert!(mk("http://127.0.0.1:1/v1", Locality::Device).is_none());
        assert!(mk("http://192.168.50.205:11434", Locality::Home).is_none());
        assert!(mk("http://studio.local:11434", Locality::Home).is_none());
        assert_eq!(
            mk("http://8.8.8.8:11434", Locality::Home),
            Some(ConfigError::HomeMustBePrivate)
        );
        assert_eq!(mk("not a url", Locality::Device), Some(ConfigError::BadUrl));
        // a host that merely CONTAINS "localhost" is not loopback
        assert_eq!(
            mk("http://localhost.evil.example", Locality::Device),
            Some(ConfigError::DeviceMustBeLoopback)
        );
        // user-info tricks: the real host is after the @
        assert_eq!(
            mk("http://127.0.0.1@evil.example/", Locality::Device),
            Some(ConfigError::DeviceMustBeLoopback)
        );
    }

    #[test]
    fn a_remote_provider_must_come_with_a_credential_and_a_budget() {
        let mut c = OpenAiCompatConfig::ollama("m");
        c.base_url = "https://api.example.com".into();
        c.locality = Locality::Saas;
        assert_eq!(
            OpenAiCompatAdapter::new(c.clone(), None, None).err(),
            Some(ConfigError::SaasNeedsCredentialAndBudget)
        );
        c.credential_name = Some("key".into());
        let b = || {
            Some(Budget {
                ledger: Arc::new(Ledger::new(1_000_000)),
                pricing: Pricing {
                    input_micro_per_mtok: 1_000_000,
                    output_micro_per_mtok: 2_000_000,
                    ..Pricing::default()
                },
            })
        };
        assert_eq!(
            OpenAiCompatAdapter::new(
                c.clone(),
                Some(Arc::new(InMemoryCredentials::default())),
                b()
            )
            .err(),
            Some(ConfigError::CredentialUnavailable)
        );
        assert!(OpenAiCompatAdapter::new(
            c.clone(),
            Some(Arc::new(InMemoryCredentials::with("key", "sk-test"))),
            b()
        )
        .is_ok());
        c.locality = Locality::Unspecified;
        assert_eq!(
            OpenAiCompatAdapter::new(c, None, None).err(),
            Some(ConfigError::SaasNeedsCredentialAndBudget),
            "an unspecified locality is treated as remote"
        );
    }

    #[tokio::test]
    async fn a_credential_is_sent_as_a_bearer_token_and_the_remote_budget_is_enforced() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-live-TEST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_ok("paid", "stop")))
            .expect(1)
            .mount(&s)
            .await;
        let mut c = OpenAiCompatConfig::ollama("m");
        c.base_url = s.uri();
        c.locality = Locality::Saas; // (the mock listens on loopback; locality is a declaration for SaaS, address rules apply to Device/Home only)
        c.credential_name = Some("key".into());
        let ledger = Arc::new(Ledger::new(10_000_000));
        let pricing = Pricing {
            input_micro_per_mtok: 1_000_000,
            output_micro_per_mtok: 2_000_000,
            ..Pricing::default()
        };
        let a = OpenAiCompatAdapter::new(
            c.clone(),
            Some(Arc::new(InMemoryCredentials::with("key", "sk-live-TEST"))),
            Some(Budget {
                ledger: ledger.clone(),
                pricing,
            }),
        )
        .unwrap();
        assert_eq!(out(&a.call(&text("hello")).await.unwrap()), "paid");
        assert_eq!(
            ledger.spent(),
            pricing.cost(&Usage {
                input_tokens: 7,
                output_tokens: 3,
                ..Usage::default()
            })
        );
        assert!(!format!("{a:?}").contains("sk-live"));
        // a budget that cannot cover the worst case: nothing is sent
        let s2 = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_ok("x", "stop")))
            .expect(0)
            .mount(&s2)
            .await;
        c.base_url = s2.uri();
        let tiny = Arc::new(Ledger::new(1));
        let a2 = OpenAiCompatAdapter::new(
            c,
            Some(Arc::new(InMemoryCredentials::with("key", "sk-live-TEST"))),
            Some(Budget {
                ledger: tiny,
                pricing,
            }),
        )
        .unwrap();
        assert_eq!(
            a2.call(&text("hello")).await.unwrap_err(),
            ProviderError::BudgetExceeded
        );
    }

    #[tokio::test]
    async fn embeddings_are_mapped_checked_and_normalised_on_request() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(body_partial_json(json!({"model":"embed-model","input":["a","b"]})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"model":"embed-model","data":[{"embedding":[3.0,4.0]},{"embedding":[0.0,2.0]}],"usage":{"prompt_tokens":4}})))
            .mount(&s)
            .await;
        let mut c = cfg(&s);
        c.embedding_model = Some("embed-model".into());
        let a = OpenAiCompatAdapter::new(c, None, None).unwrap();
        assert_eq!(
            a.info().kinds,
            vec![RequestKind::TextGeneration, RequestKind::Embedding]
        );
        let req = |normalize| {
            Request::Embedding(EmbeddingParams {
                model_id: "x".into(),
                inputs: vec!["a".into(), "b".into()],
                normalize,
                session_id: None,
                request_id: None,
            })
        };
        match a.call(&req(true)).await.unwrap().output {
            Output::Embeddings(e) => {
                assert_eq!((e.vectors.len(), e.dimensions), (2, 2));
                assert!(
                    (e.vectors[0][0] - 0.6).abs() < 1e-6 && (e.vectors[0][1] - 0.8).abs() < 1e-6,
                    "unit length: {:?}",
                    e.vectors[0]
                );
            }
            o => panic!("{o:?}"),
        }
        match a.call(&req(false)).await.unwrap().output {
            Output::Embeddings(e) => assert_eq!(e.vectors[0], vec![3.0, 4.0]),
            o => panic!("{o:?}"),
        }
        // without an embedding model the adapter does not claim the kind
        let s2 = MockServer::start().await;
        assert_eq!(
            OpenAiCompatAdapter::new(cfg(&s2), None, None)
                .unwrap()
                .info()
                .kinds,
            vec![RequestKind::TextGeneration]
        );
        // a reply with the wrong number of vectors is an error, not a silently short result
        let s3 = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"data":[{"embedding":[1.0]}]})),
            )
            .mount(&s3)
            .await;
        let mut c3 = cfg(&s3);
        c3.embedding_model = Some("e".into());
        assert_eq!(
            OpenAiCompatAdapter::new(c3, None, None)
                .unwrap()
                .call(&req(false))
                .await
                .unwrap_err(),
            ProviderError::Other
        );
    }

    #[tokio::test]
    async fn model_discovery_lists_the_servers_models_with_unknown_capacity() {
        let s = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"object":"list","data":[{"id":"qwen2.5:7b"},{"id":"gemma4:12b"}]}),
            ))
            .mount(&s)
            .await;
        let a = OpenAiCompatAdapter::new(cfg(&s), None, None).unwrap();
        let cards = a.discover_models("2026-10-11").await.unwrap();
        assert_eq!(
            cards
                .iter()
                .map(|c| c.model_id.as_str())
                .collect::<Vec<_>>(),
            vec!["qwen2.5:7b", "gemma4:12b"]
        );
        assert!(
            cards
                .iter()
                .all(|c| c.max_input_tokens.is_none() && !c.fits_input(1)),
            "capacity is not in this endpoint, so it is unknown, not assumed"
        );
        assert!(a.health().await);
    }

    /// Against a REAL local Ollama. `AI_KIT_OLLAMA_MODEL` (default `qwen2.5:7b`) must be installed.
    #[tokio::test]
    #[ignore = "needs a running Ollama with the model installed"]
    async fn a_real_local_ollama_answers_through_the_adapter() {
        let model = std::env::var("AI_KIT_OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".into());
        let url =
            std::env::var("AI_KIT_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
        let mut c = OpenAiCompatConfig::ollama(&model);
        c.base_url = url;
        let a = OpenAiCompatAdapter::new(c, None, None).unwrap();
        assert!(a.health().await, "Ollama is not answering");
        let cards = a.discover_models("2026-10-11").await.unwrap();
        assert!(
            cards.iter().any(|c| c.model_id == model),
            "model {model} not installed: {:?}",
            cards.iter().map(|c| &c.model_id).collect::<Vec<_>>()
        );
        let r = a
            .call(&text("Reply with exactly the single word: pong"))
            .await
            .unwrap();
        println!(
            "served by {}: {:?} ({:?}), {} bytes sent",
            r.model_served,
            out(&r),
            r.finish,
            r.bytes_sent
        );
        assert!(out(&r).to_lowercase().contains("pong"), "{:?}", out(&r));
    }

    /// The whole stack, live: a real local Ollama and a (scripted) cloud provider behind the planner. Private content with no egress rule must be
    /// answered locally, the cloud must never be called, and the plan must say why it was left out.
    #[tokio::test]
    #[ignore = "needs a running Ollama with the model installed"]
    async fn live_private_content_is_answered_locally_through_the_planner_and_the_cloud_is_never_called(
    ) {
        use crate::egress::Locality;
        use crate::planner::{Entry, Filter, PlanRequest, ProviderSet};
        use crate::provider::ScriptedProvider;
        let model = std::env::var("AI_KIT_OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".into());
        let mut c = OpenAiCompatConfig::ollama(&model);
        c.base_url =
            std::env::var("AI_KIT_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
        let ollama = Arc::new(OpenAiCompatAdapter::new(c, None, None).unwrap());
        let mut cards = ollama.discover_models("2026-10-11").await.unwrap();
        let mut card = cards
            .drain(..)
            .find(|c| c.model_id == model)
            .expect("model not installed");
        card.max_input_tokens = Some(8192); // Ollama's model list does not say; the owner declares it
        let cloud = ScriptedProvider::new("cloud", Locality::Saas, &[RequestKind::TextGeneration]);
        let mut set = ProviderSet::new();
        set.add(
            Entry::new(
                cloud.clone(),
                ModelCard {
                    provider_id: "cloud".into(),
                    model_id: "cloud-model".into(),
                    ..card.clone()
                },
            )
            .with_prior(0.99),
        );
        set.add(Entry::new(ollama, card).with_prior(0.5));

        let label = Label::new(
            SourceKind::HouseholdFile,
            &["health"],
            &[Subject::person("kid", AgeBand::Minor)],
        );
        let mut pr = PlanRequest::new(RequestKind::TextGeneration, &label, Purpose::Answer, 1);
        pr.input_tokens = 20;
        pr.max_output_tokens = 20;
        let sink = VecSink::default();
        let (result, record, plan) = set
            .run(
                &pr,
                &text("Reply with exactly the single word: pong"),
                &Policy::default(),
                &sink,
            )
            .await;
        let reply = result.expect("the local model should answer");
        println!(
            "plan: {:?}; answered by {}: {:?}",
            plan.hops.iter().map(|h| &h.provider_id).collect::<Vec<_>>(),
            reply.model_served,
            out(&reply)
        );
        assert!(out(&reply).to_lowercase().contains("pong"));
        assert_eq!(
            cloud.calls(),
            0,
            "private content with no egress rule must never reach the cloud"
        );
        assert!(
            plan.rejected
                .iter()
                .any(|r| r.provider_id == "cloud" && r.filter == Filter::Egress),
            "{:?}",
            plan.rejected
        );
        assert!(record.attempts.iter().all(|a| !a.locality.leaves_home()));
        assert!(
            sink.0.lock().unwrap().is_empty(),
            "a purely local answer needs no egress audit record"
        );
    }

    // ---- the ledger fails closed ----

    const PAID: Pricing = Pricing {
        input_micro_per_mtok: 1_000_000,
        output_micro_per_mtok: 2_000_000,
        cache_read_micro_per_mtok: 0,
        cache_write_micro_per_mtok: 0,
    };

    fn saas(
        s: &MockServer,
        ledger: &Arc<Ledger>,
        pricing: Pricing,
        timeout: Duration,
    ) -> Result<OpenAiCompatAdapter, ConfigError> {
        let mut c = cfg(s);
        c.locality = Locality::Saas;
        c.credential_name = Some("key".into());
        c.embedding_model = Some("embed-model".into());
        c.timeout = timeout;
        OpenAiCompatAdapter::build(
            c,
            Some(Arc::new(InMemoryCredentials::with("key", "sk-test"))),
            Some(Budget {
                ledger: ledger.clone(),
                pricing,
            }),
            |b| b.no_proxy(), // a proxy in the environment must not decide these tests
        )
    }
    fn embed_req(inputs: &[&str]) -> Request {
        Request::Embedding(EmbeddingParams {
            model_id: "x".into(),
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            normalize: false,
            session_id: None,
            request_id: None,
        })
    }
    fn chat_worst() -> u64 {
        PAID.worst_case("hi".len() as u64 / 3 + 1, 50)
    }

    #[tokio::test]
    async fn saas_embeddings_are_reserved_and_refused_before_anything_is_sent_when_they_do_not_fit()
    {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(
                    json!({"data":[{"embedding":[1.0]}],"usage":{"prompt_tokens":1}}),
                ),
            )
            .expect(0)
            .mount(&s)
            .await;
        // one micro-dollar cannot cover the worst case of a long input
        let ledger = Arc::new(Ledger::new(1));
        let a = saas(&s, &ledger, PAID, Duration::from_secs(5)).unwrap();
        let long = "x".repeat(30_000_000);
        assert_eq!(
            a.call(&embed_req(&[&long])).await.unwrap_err(),
            ProviderError::BudgetExceeded
        );
        assert_eq!(ledger.available(), 1);
    }

    #[tokio::test]
    async fn saas_embeddings_are_settled_from_the_reported_usage() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"data":[{"embedding":[1.0]}],"usage":{"prompt_tokens":2000}}),
            ))
            .mount(&s)
            .await;
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas(&s, &ledger, PAID, Duration::from_secs(5)).unwrap();
        a.call(&embed_req(&["a"])).await.unwrap();
        assert_eq!(
            ledger.spent(),
            PAID.cost(&Usage {
                input_tokens: 2000,
                ..Usage::default()
            })
        );
        assert_eq!(ledger.available(), 10_000_000 - ledger.spent());
    }

    #[tokio::test]
    async fn a_chat_200_without_usage_is_settled_at_the_reserved_worst_case() {
        let s = MockServer::start().await;
        serve(
            &s,
            200,
            json!({"model":"m","choices":[{"message":{"content":"x"},"finish_reason":"stop"}]}),
        )
        .await;
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas(&s, &ledger, PAID, Duration::from_secs(5)).unwrap();
        a.call(&text("hi")).await.unwrap();
        assert_eq!(ledger.spent(), chat_worst());
    }

    #[tokio::test]
    async fn a_chat_timeout_after_the_request_was_sent_is_settled_at_the_reserved_worst_case() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_ok("late", "stop"))
                    .set_delay(Duration::from_secs(5)),
            )
            .mount(&s)
            .await;
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas(&s, &ledger, PAID, Duration::from_millis(200)).unwrap();
        assert_eq!(
            a.call(&text("hi")).await.unwrap_err(),
            ProviderError::Timeout
        );
        assert_eq!(ledger.spent(), chat_worst());
    }

    #[tokio::test]
    async fn a_chat_connection_that_was_never_made_is_released_at_no_charge() {
        let s = MockServer::start().await;
        let ledger = Arc::new(Ledger::new(10_000_000));
        let mut c = cfg(&s);
        c.locality = Locality::Saas;
        c.base_url = "http://127.0.0.1:1".into();
        c.credential_name = Some("key".into());
        let a = OpenAiCompatAdapter::build(
            c,
            Some(Arc::new(InMemoryCredentials::with("key", "sk-test"))),
            Some(Budget {
                ledger: ledger.clone(),
                pricing: PAID,
            }),
            |b| b.no_proxy(),
        )
        .unwrap();
        assert!(a.call(&text("hi")).await.is_err());
        assert_eq!((ledger.spent(), ledger.available()), (0, 10_000_000));
    }

    /// A remote provider config with or without an embedding model, for the construction tests.
    fn remote(
        locality: Locality,
        embeds: bool,
        pricing: Pricing,
    ) -> Result<OpenAiCompatAdapter, ConfigError> {
        let mut c = OpenAiCompatConfig::ollama("m");
        c.base_url = "https://api.example.com".into();
        c.locality = locality;
        c.credential_name = Some("key".into());
        c.embedding_model = embeds.then(|| "embed-model".to_string());
        OpenAiCompatAdapter::new(
            c,
            Some(Arc::new(InMemoryCredentials::with("key", "sk-test"))),
            Some(Budget {
                ledger: Arc::new(Ledger::new(1_000_000)),
                pricing,
            }),
        )
    }

    #[test]
    fn a_remote_provider_whose_prices_cannot_cap_every_kind_it_offers_is_refused() {
        let no_input = Pricing {
            input_micro_per_mtok: 0,
            ..PAID
        };
        let no_output = Pricing {
            output_micro_per_mtok: 0,
            ..PAID
        };
        for locality in [Locality::Saas, Locality::Unspecified] {
            for embeds in [false, true] {
                for (name, pricing) in [
                    ("all zero", Pricing::default()),
                    ("no input rate", no_input),
                    ("no output rate", no_output),
                ] {
                    assert_eq!(
                        remote(locality, embeds, pricing).err(),
                        Some(ConfigError::UnpricedProvider),
                        "{locality:?}, embeddings {embeds}, {name}"
                    );
                }
                assert!(remote(locality, embeds, PAID).is_ok());
            }
        }
    }

    #[test]
    fn a_device_provider_is_not_made_to_declare_prices() {
        let mut c = OpenAiCompatConfig::ollama("m");
        c.embedding_model = Some("embed-model".into());
        let budget = Budget {
            ledger: Arc::new(Ledger::new(1)),
            pricing: Pricing::default(),
        };
        assert!(OpenAiCompatAdapter::new(c, None, Some(budget)).is_ok());
    }

    #[tokio::test]
    async fn an_embedding_200_without_usage_is_settled_at_the_reserved_worst_case() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"data":[{"embedding":[1.0]}]})),
            )
            .mount(&s)
            .await;
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas(&s, &ledger, PAID, Duration::from_secs(5)).unwrap();
        a.call(&embed_req(&["some text to embed"])).await.unwrap();
        let reserved = Budget {
            ledger: ledger.clone(),
            pricing: PAID,
        }
        .embedding_worst_case(&["some text to embed".to_string()]);
        assert!(reserved > 0);
        assert_eq!(
            ledger.spent(),
            reserved,
            "no usage reported: the worst case"
        );
        assert_eq!(ledger.available(), 10_000_000 - reserved);
    }

    #[tokio::test]
    async fn an_embedding_200_with_usage_but_no_prompt_count_is_settled_at_the_worst_case() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(
                    json!({"data":[{"embedding":[1.0]}],"usage":{"total_tokens":1}}),
                ),
            )
            .mount(&s)
            .await;
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas(&s, &ledger, PAID, Duration::from_secs(5)).unwrap();
        a.call(&embed_req(&["some text to embed"])).await.unwrap();
        let reserved = Budget {
            ledger: ledger.clone(),
            pricing: PAID,
        }
        .embedding_worst_case(&["some text to embed".to_string()]);
        assert_eq!(ledger.spent(), reserved);
    }

    #[tokio::test]
    async fn a_chat_200_with_only_part_of_the_usage_is_settled_at_the_reserved_worst_case() {
        for usage in [json!({"prompt_tokens": 1}), json!({"completion_tokens": 1})] {
            let s = MockServer::start().await;
            serve(
                &s,
                200,
                json!({"model":"m","choices":[{"message":{"content":"x"},"finish_reason":"stop"}],"usage":usage}),
            )
            .await;
            let ledger = Arc::new(Ledger::new(10_000_000));
            let a = saas(&s, &ledger, PAID, Duration::from_secs(5)).unwrap();
            a.call(&text("hi")).await.unwrap();
            assert_eq!(ledger.spent(), chat_worst(), "usage {usage}");
        }
    }

    #[tokio::test]
    async fn a_connect_timeout_is_a_request_never_sent_and_is_released_at_no_charge() {
        let Some(addr) = super::super::test_support::blackholed_addr() else {
            return;
        };
        let s = MockServer::start().await;
        let ledger = Arc::new(Ledger::new(10_000_000));
        let mut c = cfg(&s);
        c.locality = Locality::Saas;
        c.base_url = format!("http://{addr}");
        c.credential_name = Some("key".into());
        c.timeout = Duration::from_secs(20);
        c.connect_timeout = Duration::from_millis(300);
        let a = OpenAiCompatAdapter::build(
            c,
            Some(Arc::new(InMemoryCredentials::with("key", "sk-test"))),
            Some(Budget {
                ledger: ledger.clone(),
                pricing: PAID,
            }),
            |b| b.no_proxy(),
        )
        .unwrap();
        let started = std::time::Instant::now();
        assert!(a.call(&text("hi")).await.is_err());
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the connect limit, not the overall one, ended the call"
        );
        assert_eq!((ledger.spent(), ledger.available()), (0, 10_000_000));
    }

    fn saas_at(
        base: &str,
        creds: Arc<dyn CredentialSource>,
        ledger: &Arc<Ledger>,
        timeout: Duration,
        connect: Duration,
    ) -> OpenAiCompatAdapter {
        let mut c = OpenAiCompatConfig::ollama("test-model");
        c.base_url = base.into();
        c.locality = Locality::Saas;
        c.credential_name = Some("key".into());
        c.embedding_model = Some("embed-model".into());
        c.timeout = timeout;
        c.connect_timeout = connect;
        OpenAiCompatAdapter::build(
            c,
            Some(creds),
            Some(Budget {
                ledger: ledger.clone(),
                pricing: PAID,
            }),
            |b| b.no_proxy(),
        )
        .unwrap()
    }
    fn key_source() -> Arc<dyn CredentialSource> {
        Arc::new(InMemoryCredentials::with("key", "sk-test"))
    }
    fn embed_worst(ledger: &Arc<Ledger>, inputs: &[&str]) -> u64 {
        Budget {
            ledger: ledger.clone(),
            pricing: PAID,
        }
        .embedding_worst_case(&inputs.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[tokio::test]
    async fn a_reply_that_breaks_while_its_body_is_read_is_charged_for_chat_and_embeddings() {
        let base = format!(
            "http://{}",
            super::super::test_support::truncated_reply_server()
        );
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas_at(
            &base,
            key_source(),
            &ledger,
            Duration::from_secs(5),
            Duration::from_secs(1),
        );
        assert!(a.call(&text("hi")).await.is_err());
        assert_eq!(ledger.spent(), chat_worst());
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas_at(
            &base,
            key_source(),
            &ledger,
            Duration::from_secs(5),
            Duration::from_secs(1),
        );
        assert!(a.call(&embed_req(&["hi"])).await.is_err());
        assert_eq!(ledger.spent(), embed_worst(&ledger, &["hi"]));
    }

    #[tokio::test]
    async fn an_error_status_is_released_unless_it_is_one_a_gateway_uses_for_giving_up() {
        for (status, charged) in [
            (408u16, true),
            (502, true),
            (504, true),
            (400, false),
            (401, false),
            (429, false),
            (500, false),
            (503, false),
        ] {
            for embed in [false, true] {
                let s = MockServer::start().await;
                Mock::given(method("POST"))
                    .respond_with(
                        ResponseTemplate::new(status)
                            .set_body_json(json!({"error":{"message":"x"}})),
                    )
                    .mount(&s)
                    .await;
                let ledger = Arc::new(Ledger::new(10_000_000));
                let a = saas_at(
                    &s.uri(),
                    key_source(),
                    &ledger,
                    Duration::from_secs(5),
                    Duration::from_secs(1),
                );
                let r = if embed {
                    embed_req(&["hi"])
                } else {
                    text("hi")
                };
                assert!(a.call(&r).await.is_err());
                let worst = if embed {
                    embed_worst(&ledger, &["hi"])
                } else {
                    chat_worst()
                };
                let want = if charged { worst } else { 0 };
                assert_eq!(ledger.spent(), want, "HTTP {status}, embeddings {embed}");
                assert_eq!(
                    ledger.available(),
                    10_000_000 - want,
                    "HTTP {status}, embeddings {embed}"
                );
            }
        }
    }

    #[tokio::test]
    async fn an_embedding_that_timed_out_after_it_was_sent_is_charged_and_one_never_connected_is_released(
    ) {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
            .mount(&s)
            .await;
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas_at(
            &s.uri(),
            key_source(),
            &ledger,
            Duration::from_millis(300),
            Duration::from_millis(100),
        );
        assert_eq!(
            a.call(&embed_req(&["hi"])).await.unwrap_err(),
            ProviderError::Timeout
        );
        assert_eq!(ledger.spent(), embed_worst(&ledger, &["hi"]));

        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas_at(
            "http://127.0.0.1:1",
            key_source(),
            &ledger,
            Duration::from_secs(5),
            Duration::from_secs(1),
        );
        assert!(a.call(&embed_req(&["hi"])).await.is_err());
        assert_eq!((ledger.spent(), ledger.available()), (0, 10_000_000));
    }

    #[tokio::test]
    async fn a_credential_that_vanishes_after_the_reservation_releases_it() {
        let s = MockServer::start().await;
        let creds = super::super::test_support::Vanishing::new("sk-test");
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas_at(
            &s.uri(),
            creds.clone(),
            &ledger,
            Duration::from_secs(5),
            Duration::from_secs(1),
        );
        creds.vanish();
        assert_eq!(a.call(&text("hi")).await.unwrap_err(), ProviderError::Auth);
        assert_eq!((ledger.spent(), ledger.available()), (0, 10_000_000));
        assert_eq!(
            a.call(&embed_req(&["hi"])).await.unwrap_err(),
            ProviderError::Auth
        );
        assert_eq!((ledger.spent(), ledger.available()), (0, 10_000_000));
    }

    #[tokio::test]
    async fn a_connect_limit_longer_than_the_timeout_still_releases_a_hung_connection() {
        let Some(addr) = super::super::test_support::blackholed_addr() else {
            return;
        };
        let ledger = Arc::new(Ledger::new(10_000_000));
        let a = saas_at(
            &format!("http://{addr}"),
            key_source(),
            &ledger,
            Duration::from_millis(400),
            Duration::from_secs(30),
        );
        assert!(a.call(&text("hi")).await.is_err());
        assert_eq!((ledger.spent(), ledger.available()), (0, 10_000_000));
    }

    #[test]
    fn the_default_connect_timeout_is_applied() {
        assert_eq!(
            OpenAiCompatConfig::ollama("m").connect_timeout,
            Duration::from_secs(10)
        );
    }
}
