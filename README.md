# Inference Kit

**Local-first inference orchestration: model registry, routing, egress control and a cost ledger, with optional Candle backends.**

Inference Kit gives an application one inference API over local models and, only when you opt in, cloud providers. The default build is local-only and sends nothing anywhere. Models are immutable, hash-verified resources; every request that could leave the machine passes an explicit privacy gate first.

## Package and library names

| What | Name |
|---|---|
| Cargo package | `slashbuilder-inference-kit` |
| Library (`use ...`) | `inference_kit` |
| Command-line binary | `inference-kit` |
| Environment variables | `INFERENCE_KIT_*` |

Version 0.2.0 is the first release under these names. It replaces the earlier names (package `ai-kit`, library `ai_kit`, binary `ai-kit-cli`, `AiKitError`, `AI_KIT_*`), which remain as **deprecated** aliases for source and environment compatibility: `AiKitError` is a deprecated alias of `Error`, and each `AI_KIT_*` variable is still read when its `INFERENCE_KIT_*` counterpart is unset (a warning is logged). The aliases will be removed in a later release.

## Status

**Pre-1.0.** The API is still moving; read `CHANGELOG.md` before upgrading.

- Core trait definition and model registry (YAML catalog loading)
- Candle integration, **local dev POC** (see "Local Dev Inference (POC)" below)
- NVIDIA GPU (CUDA) support for the Candle backend: opt-in `cuda` feature, **POC** (see "GPU (CUDA) — POC" below)
- Egress router, credential sources, spend ledger and provider adapters: **opt-in cloud features** (see "Cloud features and data egress" below)
- Not yet: message transport and storage integration

## Design Principles

1. **Models are OS resources** — shipped signed in the image, immutable post-deploy.
2. **No weights over network (MVP)** — local-only; cloud fallback is v0.2+ feature.
3. **Trait-based API** — dual-layer (Rust trait + Message Kit binding).
4. **KV cache isolation** — per-session, never reused cross-app.
5. **Audit trail immutable** — kernel audit daemon or separate store.
6. **Score Kit gates capability** — authorization at call boundary.
7. **Versions lock at install** — no auto-upgrade; rebuild required.
8. **Inference-only** — no training, fine-tuning, or backprop.

## Quick Start

```rust
use inference_kit::{InferenceService, InferenceParams};

let service: Box<dyn InferenceService> = todo!(); // Bind to Candle/TFLite

let params = InferenceParams {
    model_id: "llama2-7b".to_string(),
    prompt: "Summarize this: [...]".to_string(),
    temperature: 0.7,
    top_p: 0.95,
    max_tokens: 200,
    session_id: None,
    preferred_quantization: None,
    request_id: None,
};

let response = service.infer(params).await?;
println!("Completion: {}", response.completion);
```

## Local Dev Inference (POC)

**This is a deliberate, documented exception to invariant #2** ("no weights
over network"). Production models ship signed in the OS image and are never
fetched at runtime. For local development — testing the trait, the registry,
the cache, and now a real inference backend without BenixOS, Storage Kit, or
an OS image — `--features candle` adds a `CandleInferenceService` backend
that downloads a small quantized model from Hugging Face onto your dev
machine. That download only ever happens via `make fetch-model`, is gitignored
(`dev-models/`), and is never part of any production build or boot path.

```bash
make fetch-model                 # downloads TinyLlama-1.1B-Chat (GGUF, ~700MB) into dev-models/
cargo build --features candle
cargo run --features candle --bin inference-kit -- infer \
    --model tinyllama --prompt "What is the capital of France?"
```

Without `--features candle`, `inference-kit infer` still builds and runs — it
exits with a message telling you to rebuild with the feature, rather than
failing to compile. `inference-kit hash <path>` and `cache-stats` work either way.

**Known POC limitations** (see `AGENT.md` for the full list): each `infer()`
call runs a fresh generation — there's no cross-call transformer KV-cache
reuse at the Candle level (`KvCacheManager` still tracks session bookkeeping/
TTL, just not real tensor state); CPU-only, no CUDA/Metal; single small model,
not the full model catalog / quantization-fallback story described below.

## GPU (CUDA) — POC

The Candle backend runs on an NVIDIA GPU behind the opt-in `cuda` feature. Nothing changes without it: the default build, CI and `--features candle` stay CPU-only.

```bash
# One-time, only if the machine has just the NVIDIA driver (no CUDA toolkit, no root):
./scripts/cuda-userspace-toolkit.sh            # assembles nvcc + cuBLAS/cuRAND/NVRTC from pip wheels
source ~/.local/cuda-13.2-pip/env.sh           # (skip both lines if `nvcc` is already on your PATH)

cargo build --release --features cuda          # ~6 min first time (compiles Candle's CUDA kernels)
./target/release/inference-kit devices            # cpu, cuda:0 NVIDIA GeForce RTX 3060, 11.6/12.5 GB free
./target/release/inference-kit infer --model tinyllama --prompt "Say hello" --device cuda
./target/release/inference-kit bench --model tinyllama --prompt-file p.txt --device cuda --max-tokens 64
```

`--device` (or `INFERENCE_KIT_DEVICE`) takes `cpu` (default), `cuda`, `cuda:N` or `auto`. In code:
`CandleInferenceService::with_device(catalog, DevicePreference::Cuda(0))`; `device_of(model_id)` says where a loaded model is running.

**It fails hard, it does not fall back.** An explicit `cuda`/`cuda:N` that is missing, or not compiled in, makes
`load_model` return an error that says what to fix. A model that does not fit in the GPU's *free* memory (file size + 15%,
at least 384 MB, and never below the catalog's `vram_required_mb`) returns `InferenceOom` **before the file is read**; there is no spill
to the CPU and no smaller quantisation (invariant "fail hard on OOM"). Only `auto` chooses (CUDA 0 if usable, else CPU), and the
choice stays visible through `device_of`.

**Measured** (RTX 3060 12 GB, Ryzen 5 2400G, Llama-3.2-3B Q4_K_M, greedy, model loaded once and warmed up; `inference-kit bench`):

| | CPU (`--device cpu`) | GPU (`--device cuda`) | llama.cpp on the same GPU (Ollama) |
|---|---|---|---|
| prompt processing, 244 tokens | 3 tok/s | **1,630 tok/s** | |
| prompt processing, 1,531 tokens | ~3 tok/s | **1,345 tok/s** | ~4,300 tok/s |
| decode, short context | 2.3 tok/s | **89 tok/s** | ~108 tok/s |
| decode, 1,531-token context | | **43 tok/s** | ~100 tok/s |
| model load (read + hash + parse + upload) | 6.5 s | 5 s | |

So about 40x faster decoding and 500x faster prompt processing than the CPU, and 80-85% of llama.cpp's decode speed at short context, falling to 30-45% at 1.5k tokens of context.
The very first request on a machine also pays a one-time kernel JIT (~12 s here); the driver caches it on disk afterwards.

**Why it is slower than llama.cpp** (measured with a standalone benchmark against Candle 0.9.2, same model and GPU): the gap is small for a short
prompt and grows with context, because `quantized_llama` (a) appends to the KV cache with `Tensor::cat` every token, copying the whole cache, and
(b) expands the grouped K/V heads with `repeat_kv`, in F32, with no fused attention. Those two cost about 2 ms per token at 128 tokens of context and 12 ms at
1,531, which is most of the slowdown from 94 to 41 tok/s; llama.cpp's decode barely moves (110 to ~100 tok/s). Prompt processing falls with length for
the same reason (the attention scores are materialised and the weights are dequantised to f16 for cuBLAS). Sampling with `top_p` (what `ask` uses) copies and sorts all
128,256 logits on the CPU every token and roughly halves decode speed at short context; greedy decoding does not. Fixing these means changing the vendored model code, not Candle's CUDA kernels.

**What changed in the backend to make this sensible.** The CPU POC rebuilt the whole model from raw bytes on every `infer()`;
on a GPU that would re-upload the weights to VRAM per request. The weights are now built once at `load_model` and reused, one
request at a time per model. Candle clears its KV cache whenever a forward pass starts at position 0, which every request does;
`tests/candle_device.rs` runs request A, B, A on one loaded model and checks A is unchanged and B equals a fresh service's answer, on both CPU and GPU.
`load_model` is also now idempotent (a second call is a no-op instead of a second copy in VRAM).

**Still not done** (read before relying on it):
- One GPU, one model at a time per device: there is no accounting of VRAM across several loaded models, so loading a second can fail with `InferenceOom`.
- Requests for one model are serialised; there is no batching and no GPU test in CI (the Jenkins agents have no GPU), so the feature is lint-checked on a dev box only.
- Still the POC's `quantized_llama` loader, so only plain-Llama GGUF models load (not Qwen3.5+, Gemma 4, Mistral 3). GPU support does not change that.
- A finished request's key/value tensors stay in VRAM until the next request overwrites them. Needs a security review before this serves more than one principal (invariant: KV isolation per session).
- The CPU POC's other limits remain: one chat template ("zephyr"), end-of-sequence looked up as `</s>`, fixed sampler seed.

## Hybrid local/cloud routing — POC (not ruled on)

`HybridInferenceService` (`src/hybrid.rs`) wraps a local `InferenceService` and an optional cloud one behind a `Router`. It is the *mechanism* for the
locked "local-first with cloud fallback; never local-only, never cloud-first" invariant. Cloud use itself is deferred (v0.2) and the Claude proxy is unruled,
so this is a dev POC: nothing talks to a network from this module, the cloud side is optional, and the safe behaviour is the default.

**Rules, in order.** (1) *Privacy gate, fail closed:* a request is `Private` unless the caller says `Shareable`; private content never reaches the cloud, not first,
not as an escalation, not as a fallback after a local error. (2) *Capability:* raster images are not text-in/text-out, so the router `Delegate`s them (Inference Kit has no image
capability); SVG is text and routes like code. (3) *Fit:* a prompt longer than the local context goes to the cloud if allowed. (4) *Policy:* a per-task-class table says which
tier goes first and whether to **cascade**: run locally, check the answer with a `Verifier` (`Json`, `NonEmpty`, or your own: run the tests, check the length), and pay for the
cloud only if the check fails. The plain `infer()` path has no hints, so it is local-only.

```bash
inference-kit route --explain --batch requests.jsonl --cloud-cmd x --policy policy.yaml       # decisions only, loads nothing
inference-kit route --batch requests.jsonl --device cuda --cloud-cmd 'claude -p --model haiku ...'   # run (cloud = any command reading stdin; dev only)
```

**What it bought, measured** (76 tasks, real recorded outcomes for three local and three Claude tiers, replayed under each policy):

| local tier | all local | all Claude | tasks private, cascade with cheap verifiers | same, with tests as verifier |
|---|---|---|---|---|
| 3B (the only model Candle loads) | 71% | 99% (76 cloud calls) | 84% (10 calls) | 91% (15 calls; 25 with a class table) |
| 12B | 95% | 99% | **97% (2 calls)** | 99% (3 calls) |

With the 3B tier the router matters a great deal (code 2/10, hard code 1/6, usable SVG 0/3 locally); with a 12B-class tier almost everything stays local and the cloud is a safety net for ~3% of work.
Making household data private costs the 3B tier 8 points of accuracy and the 12B tier almost none. A cascade only catches *detectable* failures: a JSON check cannot see a wrong label.

**Known limits.** The classifier is a few keyword rules (81/82 on the study prompts, which it was written against, so optimistic). Verifiers check constraints, not quality. Cloud use is counted in calls and tokens, not priced.
No audit wiring yet (`RouteTrace` carries the data). Raster images are only *delegated*; there is no image service.

## Any kind of request — multimodal (POC)

`InferenceService` stays text-in/text-out and frozen. Next to it, `src/modality.rs` defines a general contract: a `Request` / `Output` envelope for **five request kinds**
(`RequestKind`: text generation, image generation, image understanding, embeddings, speech-to-text), a `RequestService` trait ("I can do these kinds; give me a request"), and `TextService`
to lift any existing text backend into it. `MultiHybridService` (`src/routed.rs`) routes any of them across a local and a cloud tier by **which tier has the kind**,
then by the same privacy, fit and policy rules as the text router. `RequestKind`, `Request`, `Output` and `TaskClass` are `#[non_exhaustive]`: a kind added later is a compile-time prompt, not a silent default.

"Ready for any kind of request" means: every kind is representable and validated; a request nobody can serve gets a typed `UnsupportedKind` with the reason (never a
wrong-kind-of-model answer); a request only the cloud could serve, with private content, is `RequiresShareable` rather than leaked; an out-of-memory error stays an `InferenceOom` all the way out
(`Error::to_inference_error_code()` gives the wire code; the old `to_error_code()` is deprecated because its numbers collide with the fabric error space).

**Grants and re-typing.** `RoutedRequest::with_granted(&[kinds])` carries the verdict the Message Kit boundary reached (Score Kit grants): a kind that was not granted is refused with `CapabilityDenied`
before anything runs. Treating free text that is really a picture request ("paint me a watercolor of a lighthouse") as an image request is **opt-in** (`MultiHybridService::with_retyping()`), and
never happens when image generation is not in the granted set, so a text-only caller cannot get image generation by phrasing. The CLI is a trusted in-process caller and enables it (`--no-retype` to turn off).

| request kind | contract | local backend today | cloud tier today |
|---|---|---|---|
| text generation | yes | Candle, plain-Llama GGUF (CPU/CUDA) | Claude (dev adapter; proxy unruled) |
| **image generation** | yes | **Candle: `z_image` (Z-Image-Turbo, Apache-2.0, the default) and `sd_turbo` (development only)** | none (Claude cannot make raster images) |
| **embeddings** | yes | **Candle BERT, e.g. all-MiniLM-L6-v2 (384-d)** | none, by design (household files are private) |
| image understanding | yes | none | none wired (a Claude backend could declare it) |
| speech-to-text | yes | none | none |

**Cargo features.** `candle` is the umbrella and means what it always did. To leave kinds out: `text`, `image` and `embedding` (each on top of `candle-base`, the shared device/VRAM/verification code); `cuda` implies `candle`.

```bash
inference-kit imagine --catalog catalog.yaml --prompt "a watercolor of a misty cottage" --seed 7 --out cottage.png --device cuda   # no --model: the catalog's default
inference-kit embed   --catalog catalog.yaml --model minilm --query "who is the vet?" --text "Trash is Tuesday" --text "The vet is Dr. Okafor"
inference-kit route   --catalog catalog.yaml --batch mixed.jsonl --local-model llama32-3b --image-model default --embed-model minilm --device cuda --cloud-cmd '...'
```
`route --batch` lines take `"kind": "text_generation" | "image_generation" | "embedding" | "speech_to_text" | "image_understanding"`. `--explain` shows every decision without loading a model.

**Multi-file models.** An image model is several files, so a catalog entry has a `files:` list and `metadata.custom.kind`; **every file is BLAKE3-verified** on load, and the VRAM check counts all of them (or uses `custom.vram_peak_mb`, the measured peak, for a model that streams its weights).

**The default image model, and the licence guard.** `ModelRegistry::default_for(kind, allow_noncommercial)` picks the model for a kind when the caller names none: a `custom.default: "true"` entry wins, otherwise the first by id.
A model whose `custom.license` starts `non-commercial` or `research` is **never** chosen unless the caller passes `--allow-noncommercial`, so the development-only `sd-turbo` cannot silently become what a product ships.

### Z-Image-Turbo (`architecture: z_image`)

A 6B diffusion transformer, a Qwen3-4B text encoder and a 16-channel VAE (Apache-2.0 on the model card; **licence review still open**). Candle 0.11 has the blocks but would need ~20 GB resident; this backend fits a 12 GB card by
(1) **streaming the text encoder one layer at a time** (`zimage_text.rs`), (2) keeping the Q4_K_M GGUF's K-quant tensors **quantized on the GPU** and dequantizing **one block at a time**, with the small dense bf16 tensors in host RAM, and (3) a **tiled VAE decode** above 768x768.
It follows the diffusers `ZImagePipeline` exactly, including that the caption's pad tokens are attended to (Candle's own `forward` masks them, which differs from diffusers by up to 28% per step; this backend has its own forward). A catalog entry has the GGUF as the main file and `text_encoder_0..2` (bf16 safetensors shards), `vae` and `tokenizer` as extra files; width and height must be multiples of 16; `guidance_scale` is ignored (Turbo).

**Measured** (RTX 3060 12 GB, weights loaded per request, 8-9 steps): 512x512 about 22 s with ~4.5 GB peak above idle; 768x448 about 24 s; 1024x1024 about 75 s with ~6.6 GB. Together with the 3B text model the demo peaks at 9.4 GB of 11.6.
Checked against diffusers on the same GGUF: one step differs by 2-8%, which is bf16 rounding (an f32 control is as far from diffusers as bf16 is). Free-running images from the same noise are the same scene with different details.
`sd-turbo` for comparison: ~1.4 s per image, much lower quality. The whole sd-turbo stack (3B + `sd-turbo` + MiniLM) peaks about 9.8 GB above idle.

**Read before relying on it.**
- **Licence:** `stabilityai/sd-turbo` is under Stability AI's non-commercial research licence (development only, guarded above). Z-Image-Turbo and its GGUF conversion, the Qwen3 encoder and the VAE still need a studio licence review before anything ships.
- Weights load per request (most of the 22 s): no residency policy yet. One image per request; no img2img, inpainting, LoRA or batching. Attention is the plain matmul-softmax form, so activation memory grows quickly with size.
- Reproducibility: the seed fixes the starting noise on every device. Z-Image uses no per-step noise, so a seed reproduces an image exactly. `sd-turbo` also seeds its sampler noise on CUDA only; on the CPU only `steps = 1` is exactly reproducible.
- No audio or vision backend yet (the contract and the honest errors are in place). Output sanity checks (`OutputCheck::Sane`) catch blank or wrong-size images and malformed embeddings, not poor quality.
- The cloud tier for non-text kinds is empty; adding one is a `RequestService` implementation.

## Cloud features and data egress

**Read this before enabling any feature in this section.** The default build (`default = []`) has no network code compiled in. Each feature below, when enabled **and** configured with a credential, sends the prompt or input and the model output to a third-party service you choose. That data leaves your machine and is governed by that provider's terms and retention, not by this crate. Even then, a call is refused unless the egress policy allows it for that request, and a request marked private never goes to a cloud provider.

| Feature | Sends data to | What is sent |
|---|---|---|
| `claude-proxy` | Anthropic (Messages API) | prompt text and the completion |
| `provider-anthropic` | Anthropic (Messages API) | prompts and outputs |
| `provider-gemini` | Google (Gemini API) | prompts, embedding inputs and outputs |
| `provider-openai-compat` | **whichever server you configure** | chat and embedding requests. A server on your own machine stays local; a hosted gateway does not |

No feature is enabled by default, and none can send anything without an explicit, non-empty credential supplied by the caller. Names of providers appear here only to say which service a feature talks to; this project is not affiliated with, endorsed by or partnered with any of them. Each feature's rustdoc repeats this warning.

### Cloud Proxy Inference — Claude/Anthropic (experimental)

**Experimental, off by default.** Sending prompt text (not model weights) to a third-party cloud API is a real data-egress decision: enable it only if you are comfortable with that provider receiving your prompts.

`--features claude-proxy` adds `ClaudeProxyInferenceService`, a second
`InferenceService` implementation that proxies `infer()` to Anthropic's
Messages API (`POST /v1/messages`) instead of running local Candle
inference — same trait, same call site, no change to caller code.

Even with the feature compiled in, this backend is inert without an
explicit API key — `ClaudeProxyInferenceService::new` fails fast on an empty
key, so enabling the feature alone cannot send anything anywhere.

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # your own key; never committed
cargo build --features claude-proxy
```

```rust
use inference_kit::{ClaudeProxyConfig, ClaudeProxyInferenceService, InferenceService};

let service = ClaudeProxyInferenceService::new(ClaudeProxyConfig {
    api_key: std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set"),
    base_url: None, // defaults to https://api.anthropic.com
})?;
```

**Known limitations** (deliberate, to fit the existing trait's single-shot
shape): no streaming, no tool use / function calling, no multi-turn
conversation — each `infer()` call is one independent request/response, and
`quantization` in the response is always the literal `"n/a (cloud proxy)"`
since there's no local weight file to quantize.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Household Hearth Cluster                               │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  App (Pi5, embedded)                                    │
│  ├─ Trait call: inference_kit::infer("llama2-7b", prompt)  │
│  └─ In-process, RO mmap, 1–10ms                        │
│                                                           │
│  ┌────────────────────────────────────────────┐        │
│  │  Inference Kit Service (Trait + Message Kit)      │        │
│  ├────────────────────────────────────────────┤        │
│  │ • Model Registry (catalog.yaml, YAML)      │        │
│  │ • Inference Engine (trait-based)           │        │
│  │ • Model Cache (RO mmap, per-model)         │        │
│  │ • KV Cache (per-session, LRU)              │        │
│  │ • Message Kit Binding (async fallback)     │        │
│  └────────────────────────────────────────────┘        │
│         ↓           ↓            ↓          ↓           │
│         │           │            │          │           │
│      ┌──┴───┐  ┌────┴──┐  ┌─────┴──┐  ┌────┴────┐     │
│      │      │  │       │  │        │  │         │      │
│   Storage Kit Score Kit Service Kit Spec Kit Device Kit │
│   (models)  (caps, rate) (metrics) (conformance) (vRAM) │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Modules

- **`types.rs`**: Core data types (ModelMetadata, InferenceParams, SessionId, etc).
- **`error.rs`**: `Error` and `Result` (`AiKitError` is a deprecated alias).
- **`env.rs`**: `INFERENCE_KIT_*` environment variables, with the deprecated `AI_KIT_*` fallback.
- **`service.rs`**: `InferenceService` trait (frozen ABI).
- **`registry.rs`**: Model discovery from OS catalog.
- **`cache.rs`**: Model cache (RO mmap) + KV cache (per-session).

## Integration Roadmap

### Q4 2026 (MVP)
- Core trait + in-process sync
- Lazy-load by default
- Candle + TFLite bindings (optional)
- Score Kit capability grants
- Service Kit observability
- Spec Kit conformance gate

### Q1 2027 (v2)
- Pre-load optimization for Cumulus
- Message Kit async fallback (Pi5 → Cumulus over Hearth)
- Quantization selection hints

### Q2+ 2027 (v3+)
- Cloud hub relay (federation)
- TPM attestation (optional)
- Fine-grained privacy controls

## Development

```bash
cargo build
cargo test
cargo fmt
cargo clippy -- -D warnings
```

## Related Repos

- **Message Kit**: Protocol envelopes for async fallback (IPC over Hearth)
- **Storage Kit**: Model artifact persistence, BLAKE3 hashing, dedup
- **Score Kit**: Capability grants (rate limiting, SLA tracking)
- **Service Kit**: Observability bridge (metrics, logging)
- **Spec Kit**: Conformance validation (model shape validation in CI)
- **Device Kit**: vRAM queries, quantization hints per hardware profile
- **Substrate Kit**: Sidecar deployment (separate process for shared inference)

## License

Apache 2.0 (SlashBuilder open infrastructure standard).
