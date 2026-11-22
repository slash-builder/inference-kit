//! `ai-kit-cli` — minimal CLI for local dev/testing.
//!
//! Hand-rolled flag parsing on purpose: a handful of flags doesn't justify
//! a new dependency (no `clap` in Cargo.toml's dependency list).
//!
//! Subcommands:
//!   hash <path>                    Print the BLAKE3 hex digest of a file.
//!                                   Always available (no `candle` feature needed) —
//!                                   this is what `scripts/fetch-model.sh` shells out to.
//!   infer --model <id> --prompt <text> [--temperature F] [--top-p F] [--max-tokens N]
//!                                   Real local inference. Requires `--features candle`
//!                                   and a fetched model (`make fetch-model`).
//!   cache-stats                    Print cache_stats() for a freshly constructed service.
//!   devices                        List the devices `--device` can name (cpu, plus any CUDA GPUs).
//!   bench ...                      Load once, warm up, then time prefill and decode separately.
//!   route ...                      Hybrid local/cloud routing (POC): explain a decision, or run requests through the
//!                                   local backend with an optional cloud command. Dev tool; see src/hybrid.rs.

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let Some(command) = args.get(1) else {
        eprint_usage();
        return ExitCode::from(2);
    };

    match command.as_str() {
        "hash" => cmd_hash(&args[2..]),
        "infer" => cmd_infer(&args[2..]),
        "cache-stats" => cmd_cache_stats(&args[2..]),
        "devices" => cmd_devices(),
        "bench" => cmd_bench(&args[2..]),
        "route" => cmd_route(&args[2..]),
        "imagine" => cmd_imagine(&args[2..]),
        "embed" => cmd_embed(&args[2..]),
        other => {
            eprintln!("error: unknown subcommand '{other}'");
            eprint_usage();
            ExitCode::from(2)
        }
    }
}

fn eprint_usage() {
    eprintln!("Usage:");
    eprintln!("  ai-kit-cli hash <path>");
    eprintln!(
        "  ai-kit-cli infer --model <id> --prompt <text> [--temperature F] [--top-p F] [--max-tokens N] [--session-id S] [--device cpu|auto|cuda|cuda:N]"
    );
    eprintln!("  ai-kit-cli cache-stats [--catalog <path>]");
    eprintln!("  ai-kit-cli devices");
    eprintln!("  ai-kit-cli route (--prompt <text> | --prompt-file <path> | --batch <jsonl>) [--explain] [--class C] [--privacy private|shareable]");
    eprintln!("                   [--verify none|nonempty|json|svg|exec|maxwords:N]  (exec runs model-generated python UNSANDBOXED) [--local-model <id>] [--device D] [--cloud-cmd <shell cmd>] [--policy <yaml>] [--max-tokens N]");
    eprintln!("  ai-kit-cli imagine --model <id> --prompt <text> --out <file.png> [--width N] [--height N] [--steps N] [--seed N] [--guidance F] [--negative <text>] [--device D]");
    eprintln!("  ai-kit-cli embed --model <id> (--text <t> ...) [--query <q>] [--device D]");
    eprintln!("  ai-kit-cli bench --model <id> (--prompt <text> | --prompt-file <path>) [--device D] [--max-tokens N] [--repeat R]");
}

fn cmd_hash(args: &[String]) -> ExitCode {
    let Some(path) = args.first() else {
        eprintln!("error: hash requires a <path> argument");
        return ExitCode::from(2);
    };
    match std::fs::read(path) {
        Ok(bytes) => {
            println!("{}", blake3::hash(&bytes).to_hex());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: failed to read {path}: {e}");
            ExitCode::FAILURE
        }
    }
}

// Only used by the candle-gated subcommands below; without the feature,
// there's nothing that needs flag parsing beyond `hash <path>`.
#[cfg(feature = "candle")]
fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

#[cfg(feature = "candle")]
fn default_catalog_path() -> String {
    env::var("AI_KIT_MODEL_DIR").unwrap_or_else(|_| "./dev-models".to_string()) + "/catalog.yaml"
}

#[cfg(feature = "candle")]
fn cmd_infer(args: &[String]) -> ExitCode {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    rt.block_on(cmd_infer_async(args))
}

#[cfg(feature = "candle")]
async fn cmd_infer_async(args: &[String]) -> ExitCode {
    use ai_kit::{CandleInferenceService, DevicePreference, InferenceParams, InferenceService};

    let Some(model_id) = flag_value(args, "--model") else {
        eprintln!("error: infer requires --model <id>");
        return ExitCode::from(2);
    };
    let Some(prompt) = flag_value(args, "--prompt") else {
        eprintln!("error: infer requires --prompt <text>");
        return ExitCode::from(2);
    };
    let temperature = flag_value(args, "--temperature")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);
    let top_p = flag_value(args, "--top-p")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.95);
    let max_tokens = flag_value(args, "--max-tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let session_id = flag_value(args, "--session-id").map(ai_kit::types::SessionId::from_string);
    let catalog_path = flag_value(args, "--catalog").unwrap_or_else(default_catalog_path);
    let device = match flag_value(args, "--device")
        .or_else(|| env::var("AI_KIT_DEVICE").ok())
        .map(|v| v.parse::<DevicePreference>())
        .transpose()
    {
        Ok(d) => d.unwrap_or_default(),
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let service = match CandleInferenceService::with_device(&catalog_path, device) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to load catalog {catalog_path}: {e}");
            eprintln!("hint: run `make fetch-model` first.");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = service.load_model(&model_id).await {
        eprintln!("error: failed to load model '{model_id}': {e}");
        return ExitCode::FAILURE;
    }

    eprintln!(
        "[device: {}]",
        service
            .device_of(&model_id)
            .unwrap_or_else(|| "unknown".to_string())
    );

    let params = InferenceParams {
        model_id,
        prompt,
        temperature,
        top_p,
        max_tokens,
        session_id,
        preferred_quantization: None,
        request_id: None,
    };

    match service.infer(params).await {
        Ok(response) => {
            println!("{}", response.completion);
            eprintln!(
                "\n[{} prompt + {} completion = {} tokens, {}ms, model {} ({})]",
                response.prompt_tokens,
                response.completion_tokens,
                response.total_tokens,
                response.inference_ms,
                response.model_version,
                response.quantization,
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: inference failed: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "candle"))]
fn cmd_infer(_args: &[String]) -> ExitCode {
    eprintln!("ai-kit-cli was built without the `candle` feature — `infer` is unavailable.");
    eprintln!("Rebuild with: cargo build --features candle");
    eprintln!("(see README.md \"Local Dev Inference (POC)\")");
    ExitCode::from(2)
}

#[cfg(feature = "candle")]
fn cmd_cache_stats(args: &[String]) -> ExitCode {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    rt.block_on(cmd_cache_stats_async(args))
}

#[cfg(feature = "candle")]
async fn cmd_cache_stats_async(args: &[String]) -> ExitCode {
    use ai_kit::{CandleInferenceService, InferenceService};

    let catalog_path = flag_value(args, "--catalog").unwrap_or_else(default_catalog_path);
    let service = match CandleInferenceService::new(&catalog_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to load catalog {catalog_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    match service.cache_stats().await {
        Ok(stats) => {
            println!("{stats:#?}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "candle"))]
fn cmd_cache_stats(_args: &[String]) -> ExitCode {
    eprintln!("ai-kit-cli was built without the `candle` feature — `cache-stats` is unavailable.");
    eprintln!("Rebuild with: cargo build --features candle");
    ExitCode::from(2)
}

#[cfg(feature = "candle")]
fn cmd_devices() -> ExitCode {
    println!("cpu");
    for line in ai_kit::describe_cuda_devices() {
        println!("{line}");
    }
    ExitCode::SUCCESS
}

#[cfg(not(feature = "candle"))]
fn cmd_devices() -> ExitCode {
    eprintln!("ai-kit-cli was built without the `candle` feature — `devices` is unavailable.");
    ExitCode::from(2)
}

#[cfg(feature = "candle")]
fn cmd_bench(args: &[String]) -> ExitCode {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    rt.block_on(cmd_bench_async(args))
}

/// Load a model once, run one untimed warm-up request (CUDA compiles its kernels
/// on first use, which is not what a long-running service pays per request), then
/// time `--repeat` rounds. Prefill is measured as a 1-token request; decode speed
/// as the extra time an `--max-tokens N` request takes beyond that.
#[cfg(feature = "candle")]
async fn cmd_bench_async(args: &[String]) -> ExitCode {
    use ai_kit::{CandleInferenceService, DevicePreference, InferenceParams, InferenceService};
    use std::time::Instant;

    let Some(model_id) = flag_value(args, "--model") else {
        eprintln!("error: bench requires --model <id>");
        return ExitCode::from(2);
    };
    let prompt = match (
        flag_value(args, "--prompt"),
        flag_value(args, "--prompt-file"),
    ) {
        (Some(p), _) => p,
        (None, Some(f)) => match std::fs::read_to_string(&f) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: cannot read {f}: {e}");
                return ExitCode::from(2);
            }
        },
        _ => {
            eprintln!("error: bench requires --prompt <text> or --prompt-file <path>");
            return ExitCode::from(2);
        }
    };
    let n: u32 = flag_value(args, "--max-tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let repeat: u32 = flag_value(args, "--repeat")
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let catalog_path = flag_value(args, "--catalog").unwrap_or_else(default_catalog_path);
    let device = match flag_value(args, "--device")
        .or_else(|| env::var("AI_KIT_DEVICE").ok())
        .map(|v| v.parse::<DevicePreference>())
        .transpose()
    {
        Ok(d) => d.unwrap_or_default(),
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let service = match CandleInferenceService::with_device(&catalog_path, device) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to load catalog {catalog_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let t = Instant::now();
    if let Err(e) = service.load_model(&model_id).await {
        eprintln!("error: failed to load model '{model_id}': {e}");
        return ExitCode::FAILURE;
    }
    let load_s = t.elapsed().as_secs_f64();
    let label = service.device_of(&model_id).unwrap_or_default();

    let mk = |max_tokens| InferenceParams {
        model_id: model_id.clone(),
        prompt: prompt.clone(),
        temperature: 0.0,
        top_p: 1.0,
        max_tokens,
        session_id: None,
        preferred_quantization: None,
        request_id: None,
    };
    let t = Instant::now();
    let warm = match service.infer(mk(n)).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: inference failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let warm_s = t.elapsed().as_secs_f64();
    println!("device        {label}");
    println!("load          {load_s:.1} s (read + hash + parse + upload)");
    println!("first request {warm_s:.1} s (includes one-time kernel setup)");
    println!(
        "prompt        {} tokens, generating up to {n}",
        warm.prompt_tokens
    );
    for i in 1..=repeat {
        let a = match service.infer(mk(1)).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: prefill request failed in round {i}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let b = match service.infer(mk(n)).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: decode request failed in round {i}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let pre_ms = f64::from(a.inference_ms).max(1.0);
        let extra = f64::from(b.completion_tokens.saturating_sub(1));
        let dec_ms = (f64::from(b.inference_ms) - pre_ms).max(1.0);
        println!(
            "round {i}: prefill {:.0} ms ({:.0} tok/s) | decode {:.2} tok/s ({} tokens in {:.0} ms)",
            pre_ms,
            f64::from(a.prompt_tokens) / (pre_ms / 1000.0),
            extra / (dec_ms / 1000.0),
            extra as u32,
            dec_ms
        );
    }
    ExitCode::SUCCESS
}

#[cfg(not(feature = "candle"))]
fn cmd_bench(_args: &[String]) -> ExitCode {
    eprintln!("ai-kit-cli was built without the `candle` feature — `bench` is unavailable.");
    ExitCode::from(2)
}

// ------------------------------------------------------------------ route (hybrid local/cloud, dev POC)

#[cfg(not(feature = "candle"))]
fn cmd_route(_args: &[String]) -> ExitCode {
    eprintln!("ai-kit-cli was built without the `candle` feature — `route` is unavailable.");
    ExitCode::from(2)
}

#[cfg(feature = "candle")]
fn cmd_route(args: &[String]) -> ExitCode {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    rt.block_on(cmd_route_async(args))
}

/// Dev-only cloud tier: run a shell command with the prompt on stdin, take stdout as the completion.
/// (For this POC it is how the `claude` CLI is used, because there is no API key on the dev box.
/// A real deployment would use `ClaudeProxyInferenceService`, which is unruled; this is not that.)
#[cfg(feature = "candle")]
struct CommandCloud {
    cmd: String,
}

#[cfg(feature = "candle")]
#[async_trait::async_trait]
impl ai_kit::InferenceService for CommandCloud {
    async fn infer(&self, p: ai_kit::InferenceParams) -> ai_kit::Result<ai_kit::InferenceResponse> {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let t = std::time::Instant::now();
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(&self.cmd)
            .current_dir("/tmp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(p.prompt.as_bytes())?;
        let out = child.wait_with_output()?;
        if !out.status.success() {
            return Err(ai_kit::AiKitError::RemoteBackendError {
                backend: "command".into(),
                reason: format!("cloud command exited with {}", out.status),
            });
        }
        let completion = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let pt = (p.prompt.chars().count() / 4) as u32 + 1;
        let ct = (completion.chars().count() / 4) as u32 + 1;
        Ok(ai_kit::InferenceResponse {
            completion,
            total_tokens: pt + ct,
            prompt_tokens: pt,
            completion_tokens: ct,
            inference_ms: u32::try_from(t.elapsed().as_millis()).unwrap_or(u32::MAX),
            model_version: "command".into(),
            quantization: "cloud".into(),
            session_id: p.session_id.unwrap_or_default(),
            request_id: p.request_id.unwrap_or_default(),
        })
    }
    async fn load_model(&self, _: &str) -> ai_kit::Result<()> {
        Ok(())
    }
    async fn unload_model(&self, _: &str) -> ai_kit::Result<()> {
        Ok(())
    }
    async fn cache_stats(&self) -> ai_kit::Result<ai_kit::types::CacheStats> {
        Err(ai_kit::AiKitError::Internal {
            reason: "n/a".into(),
        })
    }
    async fn end_session(&self, _: &ai_kit::types::SessionId) -> ai_kit::Result<()> {
        Ok(())
    }
}

/// A private scratch directory for one verifier run: created fresh (never reused), readable only by the current user.
///
/// A predictable name in the shared temp directory lets another local user pre-plant a symlink there, and a plain write would follow it
/// and clobber whatever it points at. `create` fails if the path already exists (a symlink included), so a collision is retried under a
/// new name rather than followed.
#[cfg(feature = "candle")]
fn private_scratch_dir(base: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut last = None;
    for _ in 0..16 {
        let name = format!(
            "ak-verify-{}-{nanos:x}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        );
        let dir = base.join(name);
        match create_private_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("no free scratch directory name")))
}

/// Create `dir` (not its parents), mode 0700 on Unix. Fails with `AlreadyExists` if anything, a dangling symlink included, is there.
#[cfg(feature = "candle")]
fn create_private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    let mut b = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        b.mode(0o700);
    }
    b.create(dir)
}

/// Removes a scratch directory (and what is in it) when dropped, so no exit path of the verifier can leave it behind.
#[cfg(feature = "candle")]
struct ScratchDir(std::path::PathBuf);

#[cfg(feature = "candle")]
impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write `contents` to a new file `name` in `dir`: created exclusively (it must not already exist), mode 0600 on Unix.
#[cfg(feature = "candle")]
fn write_private_file(
    dir: &std::path::Path,
    name: &str,
    contents: &str,
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;
    let path = dir.join(name);
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(&path)?.write_all(contents.as_bytes())?;
    Ok(path)
}

/// `exec` verifier: pull the first ```python block (or the whole answer) out of a completion,
/// append `tests`, and run it with python3. Passing means exit code 0 within 10 seconds.
///
/// **This runs model-generated code on the host with the caller's full privileges. It is not sandboxed**: no container, no seccomp, no
/// network or filesystem restriction, no resource limit beyond the 10-second wall-clock kill. The script is kept in a private (0700)
/// directory with a 0600 file so other local users cannot read or swap it; that protects the file, not the machine. Use it only on a
/// machine and an account where running arbitrary code from the model is acceptable, never on unreviewed prompts from others.
#[cfg(feature = "candle")]
fn exec_verifier(tests: String) -> ai_kit::Verifier {
    ai_kit::Verifier::Custom(std::sync::Arc::new(move |out: &str| {
        let code = match out.split("```").nth(1) {
            Some(b) => b.strip_prefix("python").unwrap_or(b).to_string(),
            None => out.to_string(),
        };
        // The guard removes the directory on every exit path: pass, fail, timeout, a failed write or a failed spawn.
        let Ok(dir) = private_scratch_dir(&std::env::temp_dir()).map(ScratchDir) else {
            return false;
        };
        let Ok(path) = write_private_file(&dir.0, "verify.py", &format!("{code}\n{tests}\n"))
        else {
            return false;
        };
        let Ok(mut child) = std::process::Command::new("python3")
            .arg(&path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        else {
            return false;
        };
        let start = std::time::Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(st)) => break st.success(),
                Ok(None) if start.elapsed().as_secs() < 10 => {
                    std::thread::sleep(std::time::Duration::from_millis(20))
                }
                _ => {
                    // Kill, then reap so the child is not left as a zombie. Only the direct child is killed: a process it
                    // started (a shell-out, a fork) is not, and keeps running. Not fixed here; see the PR description.
                    let _ = child.kill();
                    let _ = child.wait();
                    break false;
                }
            }
        }
    }))
}

#[cfg(feature = "candle")]
fn verifier_from(spec: &str, tests: Option<&str>) -> ai_kit::Verifier {
    // `maxwords:N`: non-empty and at most N whitespace-separated words (a cheap length constraint).
    if let Some(n) = spec
        .strip_prefix("maxwords:")
        .and_then(|n| n.parse::<usize>().ok())
    {
        return ai_kit::Verifier::Custom(std::sync::Arc::new(move |o: &str| {
            let w = o.split_whitespace().count();
            w > 0 && w <= n
        }));
    }
    match spec {
        "json" => ai_kit::Verifier::Json,
        "nonempty" => ai_kit::Verifier::NonEmpty,
        // A standalone vector graphic needs both tags and the SVG namespace; without
        // `xmlns` a browser shows a broken image (the 3B model omitted it every time).
        "svg" => ai_kit::Verifier::Custom(std::sync::Arc::new(|o: &str| {
            let l = o.to_lowercase();
            l.contains("<svg")
                && l.contains("</svg>")
                && (l.contains("xmlns=\"http://www.w3.org/2000/svg\"")
                    || l.contains("xmlns='http://www.w3.org/2000/svg'"))
        })),
        "exec" => tests.map_or(ai_kit::Verifier::None, |t| exec_verifier(t.to_string())),
        _ => ai_kit::Verifier::None,
    }
}

/// One request in a `route` run (from flags or a JSONL batch line).
#[cfg(feature = "candle")]
struct RouteJob {
    id: String,
    kind: ai_kit::RequestKind,
    prompt: String,
    class: Option<ai_kit::TaskClass>,
    privacy: ai_kit::Privacy,
    verify: String,
    tests: Option<String>,
    width: u32,
    height: u32,
    steps: u32,
    seed: Option<u64>,
    inputs: Vec<String>,
    /// Audio or image file for `speech_to_text` / `image_understanding`.
    file: Option<String>,
}

/// The output check for a request: text uses the text verifiers; every other kind uses the modality's sanity check
/// (`verify: "sane"`, or any non-`none` value).
#[cfg(feature = "candle")]
fn check_for(job: &RouteJob) -> ai_kit::OutputCheck {
    use ai_kit::{Output, OutputCheck};
    // `sane` means the modality's sanity check, whatever the request is: free text that gets re-typed as an
    // image request must be checked as an image.
    if job.verify == "sane" {
        OutputCheck::Sane
    } else if job.kind == ai_kit::RequestKind::TextGeneration {
        match verifier_from(&job.verify, job.tests.as_deref()) {
            ai_kit::Verifier::None => OutputCheck::None,
            v => OutputCheck::Custom(std::sync::Arc::new(move |o: &Output| match o {
                Output::Text(t) => v.check(&t.completion).unwrap_or(true),
                _ => true,
            })),
        }
    } else if job.verify == "none" {
        OutputCheck::None
    } else {
        OutputCheck::Sane
    }
}

#[cfg(feature = "candle")]
fn parse_privacy(s: &str) -> ai_kit::Privacy {
    if s == "shareable" {
        ai_kit::Privacy::Shareable
    } else {
        ai_kit::Privacy::Private
    }
}

#[cfg(feature = "candle")]
fn device_flag(args: &[String]) -> ai_kit::DevicePreference {
    flag_value(args, "--device")
        .or_else(|| env::var("AI_KIT_DEVICE").ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or_default()
}

#[cfg(feature = "candle")]
fn job_request(job: &RouteJob) -> Result<ai_kit::Request, String> {
    use ai_kit::{
        AudioFormat, EmbeddingParams, ImageGenParams, ImageUnderstandingParams, InferenceParams,
        Request, SpeechParams,
    };
    let bytes = || -> Result<Vec<u8>, String> {
        let f = job.file.as_deref().ok_or("this kind needs a \"file\"")?;
        std::fs::read(f).map_err(|e| format!("cannot read {f}: {e}"))
    };
    Ok(match job.kind {
        ai_kit::RequestKind::ImageGeneration => {
            let mut p = ImageGenParams::new(String::new(), job.prompt.clone());
            p.width = job.width;
            p.height = job.height;
            p.steps = job.steps;
            p.seed = job.seed;
            Request::ImageGeneration(p)
        }
        ai_kit::RequestKind::Embedding => Request::Embedding(EmbeddingParams {
            model_id: String::new(),
            inputs: job.inputs.clone(),
            normalize: true,
            session_id: None,
            request_id: None,
        }),
        ai_kit::RequestKind::ImageUnderstanding => {
            let mime = match job.file.as_deref().and_then(|f| f.rsplit('.').next()) {
                Some("jpg" | "jpeg") => "image/jpeg",
                Some("webp") => "image/webp",
                _ => "image/png",
            };
            Request::ImageUnderstanding(ImageUnderstandingParams {
                model_id: String::new(),
                image: bytes()?,
                mime: mime.into(),
                prompt: job.prompt.clone(),
                max_tokens: 256,
                session_id: None,
                request_id: None,
            })
        }
        ai_kit::RequestKind::SpeechToText => Request::SpeechToText(SpeechParams {
            model_id: String::new(),
            audio: bytes()?,
            format: AudioFormat::Wav,
            language: None,
            session_id: None,
            request_id: None,
        }),
        ai_kit::RequestKind::TextGeneration => Request::Text(InferenceParams {
            model_id: String::new(),
            prompt: job.prompt.clone(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 256,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        }),
        // RequestKind is non_exhaustive: a kind added to the library later must be an explicit error here, never a text request.
        other => {
            return Err(format!(
                "this CLI cannot build a '{}' request",
                other.as_str()
            ))
        }
    })
}

#[cfg(feature = "candle")]
async fn cmd_route_async(args: &[String]) -> ExitCode {
    use ai_kit::routed::TierBackend;
    use ai_kit::{
        CandleEmbeddingService, CandleImageService, CandleInferenceService, InferenceService,
        MultiHybridService, Output, RequestKind, RoutePolicy, RoutedRequest, TaskClass,
        TextService,
    };
    use serde_json::{json, Value};
    use std::sync::Arc;

    let explain = args.iter().any(|a| a == "--explain");
    let policy = match flag_value(args, "--policy") {
        Some(f) => match std::fs::read_to_string(&f)
            .map_err(ai_kit::AiKitError::from)
            .and_then(|s| RoutePolicy::from_yaml(&s))
        {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: cannot load policy {f}: {e}");
                return ExitCode::from(2);
            }
        },
        None => RoutePolicy::local_first(),
    };
    let max_tokens: u32 = flag_value(args, "--max-tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let cloud_cmd = flag_value(args, "--cloud-cmd");
    let out_dir = flag_value(args, "--out-dir").unwrap_or_else(|| ".".into());

    // Requests: one from flags, or many from a JSONL batch (one JSON object per line).
    let mut reqs: Vec<RouteJob> = Vec::new();
    let from_json = |i: usize, v: &Value| -> Result<RouteJob, String> {
        let s = |k: &str, d: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or(d).to_string();
        let kind: RequestKind = s("kind", "text_generation").parse()?;
        let n = |k: &str, d: u64| v.get(k).and_then(|x| x.as_u64()).unwrap_or(d);
        Ok(RouteJob {
            id: s("id", &(i + 1).to_string()),
            kind,
            prompt: s("prompt", ""),
            class: v
                .get("class")
                .and_then(|c| serde_json::from_value::<TaskClass>(c.clone()).ok()),
            privacy: parse_privacy(&s("privacy", "private")),
            verify: s("verify", "none"),
            tests: v.get("tests").and_then(|x| x.as_str()).map(String::from),
            width: n("width", 512) as u32,
            height: n("height", 512) as u32,
            steps: n("steps", 2) as u32,
            seed: v.get("seed").and_then(|x| x.as_u64()),
            file: v.get("file").and_then(|x| x.as_str()).map(String::from),
            inputs: v
                .get("inputs")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|t| t.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
        })
    };
    if let Some(f) = flag_value(args, "--batch") {
        let Ok(text) = std::fs::read_to_string(&f) else {
            eprintln!("error: cannot read {f}");
            return ExitCode::from(2);
        };
        for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let parsed = serde_json::from_str::<Value>(line)
                .map_err(|e| e.to_string())
                .and_then(|v| from_json(i, &v));
            match parsed {
                Ok(j) => reqs.push(j),
                Err(e) => {
                    eprintln!("error: batch line {}: {e}", i + 1);
                    return ExitCode::from(2);
                }
            }
        }
    } else {
        let prompt = match (
            flag_value(args, "--prompt"),
            flag_value(args, "--prompt-file"),
        ) {
            (Some(p), _) => p,
            (None, Some(f)) => std::fs::read_to_string(&f).unwrap_or_default(),
            _ => {
                eprintln!("error: route needs --prompt, --prompt-file or --batch");
                return ExitCode::from(2);
            }
        };
        let v = json!({"prompt": prompt, "class": flag_value(args, "--class"), "privacy": flag_value(args, "--privacy"),
                       "verify": flag_value(args, "--verify"), "kind": flag_value(args, "--kind")});
        let mut v = v.as_object().cloned().unwrap_or_default();
        v.retain(|_, x| !x.is_null());
        let mut job = match from_json(0, &Value::Object(v)) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        };
        job.tests = flag_value(args, "--tests-file").and_then(|f| std::fs::read_to_string(f).ok());
        reqs.push(job);
    }

    // Tiers. Constructing the backends only reads catalogs: models load on first use, so --explain loads nothing.
    let catalog_path = flag_value(args, "--catalog").unwrap_or_else(default_catalog_path);
    let device = device_flag(args);
    let mut local: Vec<TierBackend> = Vec::new();
    if let Some(m) = flag_value(args, "--local-model") {
        match CandleInferenceService::with_device(&catalog_path, device) {
            Ok(svc) => {
                if !explain {
                    if let Err(e) = svc.load_model(&m).await {
                        eprintln!("error: failed to load model '{m}': {e}");
                        return ExitCode::FAILURE;
                    }
                }
                local.push(TierBackend::new(
                    Arc::new(TextService::new(Arc::new(svc))),
                    &[(RequestKind::TextGeneration, &m)],
                ));
            }
            Err(e) => {
                eprintln!("error: failed to load catalog {catalog_path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(m) = flag_value(args, "--image-model") {
        let m = match resolve_model(Some(m), RequestKind::ImageGeneration, &catalog_path, args) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        };
        match CandleImageService::new(&catalog_path, device) {
            Ok(svc) => local.push(TierBackend::new(
                Arc::new(svc),
                &[(RequestKind::ImageGeneration, &m)],
            )),
            Err(e) => {
                eprintln!("error: image catalog: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(m) = flag_value(args, "--embed-model") {
        match CandleEmbeddingService::new(&catalog_path, device) {
            Ok(svc) => local.push(TierBackend::new(
                Arc::new(svc),
                &[(RequestKind::Embedding, &m)],
            )),
            Err(e) => {
                eprintln!("error: embedding catalog: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let cloud: Vec<TierBackend> = cloud_cmd
        .map(|c| {
            let name = flag_value(args, "--cloud-model").unwrap_or_else(|| "cloud".into());
            vec![TierBackend::new(
                Arc::new(TextService::new(Arc::new(CommandCloud { cmd: c }))),
                &[(RequestKind::TextGeneration, &name)],
            )]
        })
        .unwrap_or_default();
    // The CLI is a trusted in-process caller (no boundary, so no grant set): free-text image requests may be re-typed.
    let mut svc = MultiHybridService::new(local, cloud, policy);
    if !args.iter().any(|a| a == "--no-retype") {
        svc = svc.with_retyping();
    }

    let routed = |job: &RouteJob| -> Result<RoutedRequest, String> {
        let mut r = RoutedRequest::new(job_request(job)?);
        r.class = job.class;
        r.privacy = job.privacy;
        r.check = check_for(job);
        Ok(r)
    };
    if explain {
        for job in &reqs {
            match routed(job) {
                Ok(r) => println!("{}", json!({"id": job.id, "decision": svc.explain(&r)})),
                Err(e) => println!("{}", json!({"id": job.id, "ok": false, "error": e})),
            }
        }
        return ExitCode::SUCCESS;
    }
    let _ = max_tokens;
    std::fs::create_dir_all(&out_dir).ok();
    for job in &reqs {
        let r = match routed(job) {
            Ok(r) => r,
            Err(e) => {
                println!("{}", json!({"id": job.id, "ok": false, "error": e}));
                continue;
            }
        };
        match svc.handle_routed(r).await {
            Ok((out, trace)) => {
                let body = match out {
                    Output::Text(t) => json!({"kind": "text", "completion": t.completion}),
                    Output::Image(i) => {
                        let file = format!("{out_dir}/{}.png", job.id);
                        let _ = std::fs::write(&file, &i.png);
                        json!({"kind": "image", "file": file, "width": i.width, "height": i.height, "seed": i.seed,
                               "steps": i.steps, "pixel_stddev": i.pixel_stddev, "inference_ms": i.inference_ms})
                    }
                    Output::Embeddings(e) => {
                        json!({"kind": "embeddings", "count": e.vectors.len(), "dimensions": e.dimensions,
                        "head": e.vectors.first().map(|v| v.iter().take(4).collect::<Vec<_>>()), "inference_ms": e.inference_ms})
                    }
                    Output::Transcript(t) => json!({"kind": "transcript", "text": t.text}),
                    _ => json!({"kind": "unknown", "note": "output kind newer than this CLI"}),
                };
                println!(
                    "{}",
                    json!({"id": job.id, "ok": true, "output": body, "trace": trace})
                );
            }
            Err(e) => println!(
                "{}",
                json!({"id": job.id, "ok": false, "error": e.to_string()})
            ),
        }
    }
    eprintln!("{}", json!({"stats": svc.route_stats()}));
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ imagine / embed

#[cfg(not(feature = "candle"))]
fn cmd_imagine(_args: &[String]) -> ExitCode {
    eprintln!("ai-kit-cli was built without the `candle` feature, so `imagine` is unavailable.");
    ExitCode::from(2)
}
#[cfg(not(feature = "candle"))]
fn cmd_embed(_args: &[String]) -> ExitCode {
    eprintln!("ai-kit-cli was built without the `candle` feature, so `embed` is unavailable.");
    ExitCode::from(2)
}

/// Resolve a model flag: a concrete id is used as is; `default` (or no flag, where allowed) picks the catalog's default for
/// `kind` by the registry's rule: never a non-commercial model unless `--allow-noncommercial` is given.
#[cfg(feature = "candle")]
fn resolve_model(
    given: Option<String>,
    kind: ai_kit::RequestKind,
    catalog_path: &str,
    args: &[String],
) -> Result<String, String> {
    if let Some(id) = given.filter(|g| g != "default") {
        return Ok(id);
    }
    let reg = ai_kit::registry::ModelRegistry::load_from_file(catalog_path)
        .map_err(|e| format!("failed to load catalog {catalog_path}: {e}"))?;
    let allow = args.iter().any(|a| a == "--allow-noncommercial");
    reg.default_for(kind, allow).map(|m| m.id.clone()).ok_or_else(|| {
        format!(
            "no commercially usable {} model in {catalog_path} (pass --allow-noncommercial to use a development-only model)",
            kind.as_str()
        )
    })
}

#[cfg(feature = "candle")]
fn cmd_imagine(args: &[String]) -> ExitCode {
    use ai_kit::{CandleImageService, ImageGenParams, Output, Request, RequestService};
    let (Some(prompt), Some(out)) = (flag_value(args, "--prompt"), flag_value(args, "--out"))
    else {
        eprintln!(
            "error: imagine needs --prompt and --out (and --model, or the catalog's default)"
        );
        return ExitCode::from(2);
    };
    let catalog_path = flag_value(args, "--catalog").unwrap_or_else(default_catalog_path);
    let model = match resolve_model(
        flag_value(args, "--model"),
        ai_kit::RequestKind::ImageGeneration,
        &catalog_path,
        args,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let svc = match CandleImageService::new(&catalog_path, device_flag(args)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to load catalog {catalog_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut p = ImageGenParams::new(model, prompt);
    let num = |k: &str| flag_value(args, k).and_then(|v| v.parse::<u32>().ok());
    p.width = num("--width").unwrap_or(p.width);
    p.height = num("--height").unwrap_or(p.height);
    p.steps = num("--steps").unwrap_or(p.steps);
    p.seed = flag_value(args, "--seed").and_then(|v| v.parse().ok());
    p.guidance_scale = flag_value(args, "--guidance")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    p.negative_prompt = flag_value(args, "--negative");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let t = std::time::Instant::now();
    match rt.block_on(svc.handle(Request::ImageGeneration(p))) {
        Ok(Output::Image(i)) => {
            if let Err(e) = std::fs::write(&out, &i.png) {
                eprintln!("error: cannot write {out}: {e}");
                return ExitCode::FAILURE;
            }
            println!(
                "{}",
                serde_json::json!({"file": out, "width": i.width, "height": i.height, "seed": i.seed, "steps": i.steps,
                    "pixel_stddev": i.pixel_stddev, "generate_ms": i.inference_ms, "total_ms": t.elapsed().as_millis() as u64})
            );
            ExitCode::SUCCESS
        }
        Ok(_) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "candle")]
fn cmd_embed(args: &[String]) -> ExitCode {
    use ai_kit::{CandleEmbeddingService, EmbeddingParams, Output, Request, RequestService};
    let Some(model) = flag_value(args, "--model") else {
        eprintln!("error: embed needs --model and one or more --text");
        return ExitCode::from(2);
    };
    let texts: Vec<String> = args
        .windows(2)
        .filter(|w| w[0] == "--text")
        .map(|w| w[1].clone())
        .collect();
    let query = flag_value(args, "--query");
    let mut inputs = texts.clone();
    if let Some(q) = &query {
        inputs.push(q.clone());
    }
    if inputs.is_empty() {
        eprintln!("error: embed needs at least one --text");
        return ExitCode::from(2);
    }
    let catalog_path = flag_value(args, "--catalog").unwrap_or_else(default_catalog_path);
    let svc = match CandleEmbeddingService::new(&catalog_path, device_flag(args)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to load catalog {catalog_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let req = Request::Embedding(EmbeddingParams {
        model_id: model,
        inputs,
        normalize: true,
        session_id: None,
        request_id: None,
    });
    match rt.block_on(svc.handle(req)) {
        Ok(Output::Embeddings(e)) => {
            let mut result = serde_json::json!({"count": e.vectors.len(), "dimensions": e.dimensions, "inference_ms": e.inference_ms});
            if query.is_some() {
                let q = e.vectors.last().cloned().unwrap_or_default();
                let mut ranked: Vec<(f32, &String)> = texts
                    .iter()
                    .zip(&e.vectors)
                    .map(|(t, v)| (v.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>(), t))
                    .collect();
                ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
                result["ranked"] = serde_json::json!(ranked
                    .iter()
                    .map(|(s, t)| serde_json::json!({"score": s, "text": t}))
                    .collect::<Vec<_>>());
            }
            println!("{result}");
            ExitCode::SUCCESS
        }
        Ok(_) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(all(test, feature = "candle", unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, MetadataExt};

    #[test]
    fn the_scratch_dir_is_new_private_and_never_reused() {
        let base = tempfile::tempdir().unwrap();
        let a = private_scratch_dir(base.path()).unwrap();
        let b = private_scratch_dir(base.path()).unwrap();
        assert_ne!(a, b);
        assert_eq!(std::fs::metadata(&a).unwrap().mode() & 0o777, 0o700);
        let f = write_private_file(&a, "verify.py", "print(1)\n").unwrap();
        assert_eq!(std::fs::metadata(&f).unwrap().mode() & 0o777, 0o600);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "print(1)\n");
    }

    #[test]
    fn a_pre_planted_symlink_is_refused_not_followed() {
        let base = tempfile::tempdir().unwrap();
        let victim = base.path().join("victim.txt");
        std::fs::write(&victim, "precious").unwrap();
        // a symlink where the directory would go: creation must fail rather than follow it
        let planted = base.path().join("planted");
        symlink(base.path(), &planted).unwrap();
        let e = create_private_dir(&planted).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists);
        // and a symlink where the file would go: exclusive creation refuses it
        let dir = private_scratch_dir(base.path()).unwrap();
        symlink(&victim, dir.join("verify.py")).unwrap();
        assert!(write_private_file(&dir, "verify.py", "evil").is_err());
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "precious");
    }

    /// The tests below change `TMPDIR` and `PATH`, which are process-wide, so every test in this module holds this lock (a test
    /// that only called `tempfile::tempdir()` while another had `TMPDIR` pointed at its own directory would land inside it).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Holds the lock, and puts `TMPDIR` / `PATH` back on drop.
    struct Env {
        _lock: std::sync::MutexGuard<'static, ()>,
        tmpdir: Option<std::ffi::OsString>,
        path: Option<std::ffi::OsString>,
    }

    impl Env {
        fn lock() -> Env {
            Env {
                _lock: ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner()),
                tmpdir: std::env::var_os("TMPDIR"),
                path: std::env::var_os("PATH"),
            }
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            for (k, v) in [("TMPDIR", &self.tmpdir), ("PATH", &self.path)] {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    fn python3_on_path() -> bool {
        std::process::Command::new("python3")
            .arg("-c")
            .arg("pass")
            .status()
            .is_ok_and(|s| s.success())
    }

    /// True (after saying so) when the test cannot run because there is no `python3`.
    fn skip_without_python3(test: &str) -> bool {
        let missing = !python3_on_path();
        if missing {
            eprintln!("skipping {test}: python3 is not on PATH");
        }
        missing
    }

    fn run_exec(tests: &str, out: &str) -> bool {
        let ai_kit::Verifier::Custom(f) = exec_verifier(tests.to_string()) else {
            panic!("exec is a custom verifier")
        };
        f(out)
    }

    fn entries(dir: &std::path::Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn the_exec_verifier_passes_and_fails_on_the_exit_code() {
        let _env = Env::lock();
        if skip_without_python3("the_exec_verifier_passes_and_fails_on_the_exit_code") {
            return;
        }
        let tests = "assert add(1, 2) == 3";
        assert!(run_exec(
            tests,
            "```python\ndef add(a, b):\n    return a + b\n```"
        ));
        assert!(!run_exec(
            tests,
            "```python\ndef add(a, b):\n    return a - b\n```"
        ));
    }

    /// Goes through `exec_verifier` itself: a symlink planted at the old, predictable script path must not be followed, and the
    /// run must leave nothing behind in the temp directory.
    #[test]
    fn the_exec_verifier_does_not_follow_a_planted_symlink_and_cleans_up() {
        let env = Env::lock();
        if skip_without_python3("the_exec_verifier_does_not_follow_a_planted_symlink_and_cleans_up")
        {
            return;
        }
        let victim_dir = tempfile::tempdir().unwrap();
        let victim = victim_dir.path().join("victim.txt");
        std::fs::write(&victim, "precious").unwrap();
        let tmp = tempfile::tempdir().unwrap();
        // where the previous implementation wrote: `<temp_dir>/ak-verify-<pid>.py`
        let old_path = tmp
            .path()
            .join(format!("ak-verify-{}.py", std::process::id()));
        symlink(&victim, &old_path).unwrap();
        std::env::set_var("TMPDIR", tmp.path());

        assert!(run_exec(
            "assert add(1, 2) == 3",
            "def add(a, b):\n    return a + b"
        ));

        drop(env);
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "precious");
        // only the planted link is left: no scratch directory, no script
        assert_eq!(
            entries(tmp.path()),
            vec![old_path.file_name().unwrap().to_string_lossy().into_owned()]
        );
    }

    #[test]
    fn the_scratch_directory_is_removed_when_the_interpreter_cannot_be_spawned() {
        let env = Env::lock();
        let tmp = tempfile::tempdir().unwrap();
        let empty_path = tempfile::tempdir().unwrap();
        std::env::set_var("TMPDIR", tmp.path());
        std::env::set_var("PATH", empty_path.path());

        // no python3 on this PATH, so the spawn fails; that must be a plain `false` and leave nothing behind
        assert!(!run_exec("pass", "x = 1"));

        drop(env);
        assert!(
            entries(tmp.path()).is_empty(),
            "left: {:?}",
            entries(tmp.path())
        );
    }

    #[test]
    fn the_scratch_directory_is_removed_after_a_failing_run() {
        let env = Env::lock();
        if skip_without_python3("the_scratch_directory_is_removed_after_a_failing_run") {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("TMPDIR", tmp.path());

        assert!(!run_exec("assert False", "x = 1"));

        drop(env);
        assert!(
            entries(tmp.path()).is_empty(),
            "left: {:?}",
            entries(tmp.path())
        );
    }
}
