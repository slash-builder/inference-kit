//! Inference Kit — local-first inference orchestration: model registry, routing, egress control and a cost ledger, with optional Candle backends.
//!
//! # Overview
//!
//! The package is `slashbuilder-inference-kit`; the library is `inference_kit`. The default build is
//! local-only and compiles in no network code.
//!
//! # Cloud features and data egress
//!
//! The features `claude-proxy`, `provider-anthropic`, `provider-gemini` and `provider-openai-compat` are
//! **off by default**. Each one, when enabled and configured with a credential, **sends prompts and model
//! outputs to a third-party service** (for `provider-openai-compat`, to whichever server you point it at).
//! See each module's "Data egress" section, and the README, before enabling one.
//!
//! # Core Design Principles
//!
//! 1. **Models are OS resources** — shipped signed in the image, immutable post-deploy.
//! 2. **No weights over network (MVP)** — local-only; cloud fallback is v0.2+ feature.
//! 3. **Trait-based API** — dual-layer (Rust trait + Message Kit binding).
//! 4. **KV cache isolation** — per-session, never reused cross-app.
//! 5. **Audit trail immutable** — kernel audit daemon or separate store.
//! 6. **Score Kit gates capability** — authorization at call boundary.
//! 7. **Versions lock at install** — no auto-upgrade; rebuild required.
//! 8. **Inference-only** — no training, fine-tuning, or backprop.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  Household Hearth Cluster                               │
//! ├─────────────────────────────────────────────────────────┤
//! │                                                           │
//! │  App (Pi5, embedded)                                    │
//! │  ├─ Trait call: inference_kit::infer("llama2-7b", prompt)  │
//! │  └─ In-process, RO mmap, 1–10ms                        │
//! │                                                           │
//! │  App (Cumulus, always-on)                              │
//! │  ├─ Trait call: same API                               │
//! │  └─ Or Message Kit RPC for load distribution           │
//! │                                                           │
//! │  ┌────────────────────────────────────────────┐        │
//! │  │  Inference Kit Service (Trait + Message Kit)      │        │
//! │  ├────────────────────────────────────────────┤        │
//! │  │ • Model Registry (catalog.yaml, YAML)      │        │
//! │  │ • Inference Engine (trait-based)           │        │
//! │  │ • Model Cache (RO mmap, per-model)         │        │
//! │  │ • KV Cache (per-session, LRU)              │        │
//! │  │ • Message Kit Binding (async fallback)     │        │
//! │  └────────────────────────────────────────────┘        │
//! │         ↓           ↓            ↓          ↓           │
//! │         │           │            │          │           │
//! │      ┌──┴───┐  ┌────┴──┐  ┌─────┴──┐  ┌────┴────┐     │
//! │      │      │  │       │  │        │  │         │      │
//! │   Storage Kit Score Kit Service Kit Spec Kit Device Kit │
//! │   (models)  (caps, rate) (metrics) (conformance) (vRAM) │
//! │                                                           │
//! │  /usr/share/models/ (OS-shipped, RO, signed)           │
//! │  ├─ llama2-7b-q4/ (weights.bin, meta.json)             │
//! │  ├─ whisper-tiny/ (model.bin, meta.json)               │
//! │  └─ ...                                                  │
//! │                                                           │
//! │  Audit Trail (kernel audit daemon or Service Kit)      │
//! │  ├─ Model name, device, latency, app_id, timestamp     │
//! │  └─ Never: plaintext prompts, outputs                   │
//! │                                                           │
//! └─────────────────────────────────────────────────────────┘
//! ```

#[cfg(any(feature = "candle-base", feature = "claude-proxy"))]
pub mod backends;
pub mod budget;
pub mod cache;
#[cfg(any(test, feature = "conformance"))]
pub mod conformance;
pub mod credentials;
pub mod egress;
pub mod env;
pub mod error;
pub mod hybrid;
pub mod modality;
pub mod model_card;
pub mod planner;
pub mod provider;
#[cfg(any(
    feature = "provider-anthropic",
    feature = "provider-openai-compat",
    feature = "provider-gemini"
))]
pub mod providers;
pub mod registry;
pub mod routed;
pub mod router;
pub mod service;
pub mod types;
mod yaml;

// Re-export key types at crate root
#[cfg(feature = "embedding")]
pub use backends::CandleEmbeddingService;
#[cfg(feature = "image")]
pub use backends::CandleImageService;
#[cfg(feature = "text")]
pub use backends::CandleInferenceService;
#[cfg(feature = "candle-base")]
pub use backends::{describe_cuda_devices, DevicePreference};
#[cfg(feature = "claude-proxy")]
pub use backends::{ClaudeProxyConfig, ClaudeProxyInferenceService};
/// Deprecated alias of [`Error`]; see [`error::AiKitError`].
#[allow(deprecated)]
pub use error::AiKitError;
pub use error::{Error, Result};
pub use hybrid::{
    classify, ClassPolicy, Decision, HybridInferenceService, Plan, Privacy, RoutePolicy,
    RouteRequest, RouteStats, RouteTrace, Router, TaskClass, Tier, Verifier,
};
pub use modality::{
    AudioFormat, EmbeddingParams, EmbeddingResponse, GeneratedImage, ImageGenParams,
    ImageUnderstandingParams, Output, OutputCheck, Request, RequestKind, RequestService,
    SpeechParams, TextService, TranscriptResponse,
};
pub use routed::{MultiHybridService, RoutedRequest};
pub use router::{Call, EgressRouter, EgressRouterBuilder, RouteOutcome};
pub use service::InferenceService;
pub use types::{InferenceParams, InferenceResponse, ModelFile, ModelMetadata};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
