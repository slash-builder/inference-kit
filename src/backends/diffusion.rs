//! Text-to-image on Candle: the Stable Diffusion family with the distilled "turbo" sampler. **POC.**
//!
//! [`CandleImageService`] implements [`RequestService`] for [`RequestKind::ImageGeneration`]. It is pure Rust
//! (Candle's CLIP text encoder, UNet and VAE; a Euler-ancestral scheduler; the `png` crate to encode) and runs on the CPU or, with
//! the `cuda` feature, an NVIDIA GPU, under the same device policy as the text backend: an explicit device that is
//! missing is an error, and a model that does not fit in free VRAM is an `InferenceOom` *before* any file is read.
//!
//! # What it runs
//!
//! `architecture: sd_turbo` in the catalog entry: an SD-2.1-family UNet with an OpenCLIP-H text encoder, epsilon prediction and a
//! trailing-spaced Euler-ancestral sampler (what `stabilityai/sd-turbo` ships). 512x512 in 1-4 steps. The catalog entry has the
//! UNet as the main file and `vae`, `text_encoder` and `tokenizer` as extra files; **every file is BLAKE3-verified** on load,
//! like any other model (invariant: models are immutable and verified).
//!
//! # Licence, read before shipping
//!
//! The dev weights (`stabilityai/sd-turbo`) are under Stability AI's non-commercial research licence. They are a development
//! default so the *backend* can be tested end to end. A product needs a commercially licensed model; the studio's higher-quality
//! local model (Z-Image-Turbo, Apache-2.0) cannot be run by Candle 0.11 on a 12 GB card (no quantised loader, no offload), which is why
//! it is not the default here.
//!
//! # Known limits
//!
//! Reproducibility: the starting noise comes from `seed` on every device. The Euler-ancestral sampler draws more noise on later
//! steps from Candle's own RNG, which this backend seeds on CUDA (so a seed reproduces an image exactly there) but which cannot be seeded
//! on the CPU. With `steps = 1` (no extra noise) the same seed gives the same image everywhere. No img2img, inpainting or LoRA. One request at a time per model.

use crate::backends::common::{
    check_vram, device_for, verify_model_files, DevicePreference, VerifiedFile, VerifiedMaps,
};
use crate::backends::zimage::{ZImageFiles, ZImagePipeline};
use crate::error::{AiKitError, Result};
use crate::modality::{
    unsupported, GeneratedImage, ImageGenParams, Output, Request, RequestKind, RequestService,
};
use crate::registry::ModelRegistry;
use crate::types::ModelMetadata;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::stable_diffusion::{
    clip::{ClipTextTransformer, Config as ClipConfig},
    euler_ancestral_discrete::EulerAncestralDiscreteSchedulerConfig,
    schedulers::{SchedulerConfig, TimestepSpacing},
    unet_2d::{BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig},
    vae::{AutoEncoderKL, AutoEncoderKLConfig},
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tokenizers::Tokenizer;

/// SD-2.1 family VAE scaling factor.
const VAE_SCALE: f64 = 0.18215;

struct LoadedImageModel {
    clip: ClipTextTransformer,
    clip_cfg: ClipConfig,
    unet: UNet2DConditionModel,
    vae: AutoEncoderKL,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    version: String,
}

/// One loaded image model, by architecture.
enum LoadedImage {
    SdTurbo(Box<LoadedImageModel>),
    /// Z-Image-Turbo loads its (large) weights per request, so loading it only records where the files are.
    ZImage {
        pipeline: ZImagePipeline,
        version: String,
    },
}

pub struct CandleImageService {
    registry: ModelRegistry,
    commercial_only: bool,
    device_pref: DevicePreference,
    loaded: RwLock<HashMap<String, Arc<Mutex<LoadedImage>>>>,
}

impl LoadedImage {
    fn device(&self) -> &Device {
        match self {
            LoadedImage::SdTurbo(m) => &m.device,
            LoadedImage::ZImage { pipeline, .. } => pipeline.device(),
        }
    }
}

impl CandleImageService {
    /// Refuse to load any model that is not known to permit commercial use ([`crate::types::LicenceClass::Commercial`]). Off by default;
    /// the default-model rule in the registry does not apply to a model a caller names, so a product that must ship only commercial
    /// models turns this on.
    pub fn with_commercial_only(mut self) -> Self {
        self.commercial_only = true;
        self
    }

    pub fn new(catalog_path: &str, device: DevicePreference) -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::load_from_file(catalog_path)?,
            commercial_only: false,
            device_pref: device,
            loaded: RwLock::new(HashMap::new()),
        })
    }

    /// Where a loaded model runs, or `None`.
    pub fn is_loaded(&self, model_id: &str) -> bool {
        self.loaded.read().is_ok_and(|l| l.contains_key(model_id))
    }

    fn model_meta(&self, id: &str) -> Result<ModelMetadata> {
        let m = self
            .registry
            .get(id)
            .cloned()
            .ok_or_else(|| AiKitError::ModelNotFound {
                model_id: id.to_string(),
            })?;
        if self.commercial_only {
            self.registry.check_licence(id, false)?;
        }
        if m.kind() != RequestKind::ImageGeneration {
            return Err(AiKitError::ModelLoadFailed {
                model_id: id.to_string(),
                reason: format!(
                    "'{id}' is a {} model, not an image-generation model",
                    m.kind().as_str()
                ),
            });
        }
        Ok(m)
    }

    /// Load a model (verify files, build on the device). Called on first use; call it earlier to warm up.
    pub fn load_model(&self, id: &str) -> Result<()> {
        if self.is_loaded(id) {
            return Ok(());
        }
        let model = self.model_meta(id)?;
        let fail = |reason: String| AiKitError::ModelLoadFailed {
            model_id: id.to_string(),
            reason,
        };
        let arch = model
            .metadata
            .custom
            .get("architecture")
            .map_or("sd_turbo", String::as_str);
        if !matches!(arch, "sd_turbo" | "z_image") {
            return Err(fail(format!(
                "unsupported image architecture '{arch}' (this backend runs: sd_turbo, z_image)"
            )));
        }
        let (device, _label) = device_for(self.device_pref).map_err(fail)?;
        check_vram(&device, &model)?;
        let unet_path = verify_model_files(&model)?;
        let need = |role: &str| {
            model
                .file_path(role)
                .ok_or_else(|| fail(format!("catalog entry has no '{role}' file")))
        };
        if arch == "z_image" {
            let mut shards: Vec<_> = model
                .files
                .iter()
                .filter(|f| f.role.starts_with("text_encoder"))
                .collect();
            shards.sort_by(|a, b| a.role.cmp(&b.role));
            if shards.is_empty() {
                return Err(fail("catalog entry has no 'text_encoder*' files".into()));
            }
            let strip = |l: &str| l.strip_prefix("file://").unwrap_or(l).to_string();
            let files = ZImageFiles {
                gguf: unet_path,
                text_encoder_shards: shards.iter().map(|f| strip(&f.location)).collect(),
                vae: need("vae")?,
                tokenizer: need("tokenizer")?,
            };
            self.loaded
                .write()
                .map_err(|_| AiKitError::Internal {
                    reason: "loaded-model lock poisoned".into(),
                })?
                .insert(
                    id.to_string(),
                    Arc::new(Mutex::new(LoadedImage::ZImage {
                        pipeline: ZImagePipeline::new(files, device),
                        version: model.version.clone(),
                    })),
                );
            return Ok(());
        }
        let (vae_path, clip_path, tok_path) =
            (need("vae")?, need("text_encoder")?, need("tokenizer")?);

        let dtype = if device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };
        let clip_cfg = ClipConfig::v2_1();
        // Each component is built from a handle opened once and checked against the hash taken at load (see `from_verified`), not from
        // a path Candle would open again. The tokenizer is read through the handle too.
        let tokenizer = Tokenizer::from_bytes(VerifiedFile::open(&tok_path)?.read_all()?)
            .map_err(|e| fail(format!("tokenizer: {e}")))?;
        let clip = from_verified(id, &clip_path, &device, dtype, "text encoder", |vb| {
            ClipTextTransformer::new(vb, &clip_cfg)
        })?;
        let vae = from_verified(id, &vae_path, &device, dtype, "vae", |vb| {
            AutoEncoderKL::new(vb, 3, 3, sd_vae_config())
        })?;
        let unet = from_verified(id, &unet_path, &device, dtype, "unet", |vb| {
            UNet2DConditionModel::new(vb, 4, 4, false, sd_unet_config())
        })?;
        self.loaded
            .write()
            .map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".into(),
            })?
            .insert(
                id.to_string(),
                Arc::new(Mutex::new(LoadedImage::SdTurbo(Box::new(
                    LoadedImageModel {
                        clip,
                        clip_cfg,
                        unet,
                        vae,
                        tokenizer,
                        device,
                        dtype,
                        version: model.version.clone(),
                    },
                )))),
            );
        Ok(())
    }

    fn generate(&self, p: &ImageGenParams) -> Result<GeneratedImage> {
        self.generate_inner(p).map_err(|e| {
            let device = self
                .loaded
                .read()
                .ok()
                .and_then(|l| l.get(&p.model_id).cloned())
                .and_then(|m| m.lock().ok().map(|m| m.device().clone()));
            crate::backends::common::promote_oom(e, &p.model_id, device.as_ref())
        })
    }

    fn generate_inner(&self, p: &ImageGenParams) -> Result<GeneratedImage> {
        p.validate()?;
        self.load_model(&p.model_id)?;
        let model = self
            .loaded
            .read()
            .map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".into(),
            })?
            .get(&p.model_id)
            .cloned()
            .ok_or_else(|| AiKitError::ModelNotFound {
                model_id: p.model_id.clone(),
            })?;
        let loaded = model.lock().map_err(|_| AiKitError::Internal {
            reason: "image model lock poisoned".into(),
        })?;
        let started = std::time::Instant::now();
        let seed = p.seed.unwrap_or_else(fresh_seed);
        let m = match &*loaded {
            LoadedImage::SdTurbo(m) => m,
            LoadedImage::ZImage { pipeline, version } => {
                let rgb = pipeline.generate(p, seed, None)?;
                return finish_image(rgb, p, seed, started, version);
            }
        };
        // The Euler-ancestral sampler draws fresh noise on every step but the last, from Candle's own RNG. That RNG can only be
        // seeded on CUDA, so seed it there (on the CPU only `steps = 1`, which draws none, is reproducible).
        if m.device.is_cuda() {
            m.device.set_seed(seed).map_err(|e| AiKitError::Internal {
                reason: format!("seeding the GPU RNG: {e}"),
            })?;
        }
        let internal = |what: &str, e: candle_core::Error| AiKitError::Internal {
            reason: format!("{what}: {e}"),
        };

        let (h, w) = (p.height as usize, p.width as usize);
        let use_guide = p.guidance_scale > 1.0;
        let cond = encode_prompt(m, &p.prompt).map_err(|e| internal("prompt", e))?;
        let text_embeddings = if use_guide {
            let uncond = encode_prompt(m, p.negative_prompt.as_deref().unwrap_or(""))
                .map_err(|e| internal("negative prompt", e))?;
            Tensor::cat(&[&uncond, &cond], 0).map_err(|e| internal("embeddings", e))?
        } else {
            cond
        };
        let text_embeddings = text_embeddings
            .to_dtype(m.dtype)
            .map_err(|e| internal("embeddings", e))?;

        let mut scheduler = EulerAncestralDiscreteSchedulerConfig {
            timestep_spacing: TimestepSpacing::Trailing,
            ..Default::default()
        }
        .build(p.steps as usize)
        .map_err(|e| internal("scheduler", e))?;
        let noise = seeded_normal(seed, 4 * (h / 8) * (w / 8));
        let mut latents = Tensor::from_vec(noise, (1, 4, h / 8, w / 8), &m.device)
            .and_then(|t| t * scheduler.init_noise_sigma())
            .and_then(|t| t.to_dtype(m.dtype))
            .map_err(|e| internal("latents", e))?;

        let timesteps = scheduler.timesteps().to_vec();
        for &t in &timesteps {
            let step = (|| -> candle_core::Result<Tensor> {
                let input = if use_guide {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };
                let input = scheduler.scale_model_input(input, t)?;
                let pred = m.unet.forward(&input, t as f64, &text_embeddings)?;
                let pred = if use_guide {
                    let c = pred.chunk(2, 0)?;
                    (&c[0] + ((&c[1] - &c[0])? * f64::from(p.guidance_scale))?)?
                } else {
                    pred
                };
                scheduler.step(&pred, t, &latents)
            })();
            latents = step.map_err(|e| internal("denoising step", e))?;
        }

        let rgb = (|| -> candle_core::Result<Vec<u8>> {
            let img = m.vae.decode(&(&latents / VAE_SCALE)?)?;
            let img = ((img.to_dtype(DType::F32)? / 2.)? + 0.5)?;
            let img = (img.clamp(0f32, 1f32)? * 255.)?
                .to_dtype(DType::U8)?
                .to_device(&Device::Cpu)?
                .i(0)?
                .permute((1, 2, 0))?
                .contiguous()?;
            img.flatten_all()?.to_vec1::<u8>()
        })()
        .map_err(|e| internal("decode", e))?;
        finish_image(rgb, p, seed, started, &m.version)
    }
}

/// RGB bytes -> the response (PNG, pixel spread for the sanity check, timing and ids). Shared by every architecture.
fn finish_image(
    rgb: Vec<u8>,
    p: &ImageGenParams,
    seed: u64,
    started: std::time::Instant,
    version: &str,
) -> Result<GeneratedImage> {
    let n = rgb.len() as f32;
    let mean = rgb.iter().map(|b| f32::from(*b)).sum::<f32>() / n;
    let var = rgb
        .iter()
        .map(|b| (f32::from(*b) - mean).powi(2))
        .sum::<f32>()
        / n;
    Ok(GeneratedImage {
        png: encode_png(&rgb, p.width, p.height)?,
        width: p.width,
        height: p.height,
        seed,
        steps: p.steps,
        pixel_stddev: var.sqrt(),
        inference_ms: u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX),
        model_version: version.to_string(),
        session_id: p.session_id.clone().unwrap_or_default(),
        request_id: p
            .request_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
    })
}

#[async_trait]
impl RequestService for CandleImageService {
    fn kinds(&self) -> Vec<RequestKind> {
        vec![RequestKind::ImageGeneration]
    }

    async fn handle(&self, req: Request) -> Result<Output> {
        match req {
            Request::ImageGeneration(p) => self.generate(&p).map(Output::Image),
            other => Err(unsupported(
                other.kind(),
                "this is an image-generation backend",
            )),
        }
    }
}

/// CLIP-tokenise to the model's fixed length (padded with its pad token) and run the text encoder: `(1, 77, 1024)`.
fn encode_prompt(m: &LoadedImageModel, prompt: &str) -> candle_core::Result<Tensor> {
    let max_len = m.clip_cfg.max_position_embeddings;
    let pad = m.clip_cfg.pad_with.as_deref().unwrap_or("<|endoftext|>");
    let pad_id = m
        .tokenizer
        .token_to_id(pad)
        .ok_or_else(|| candle_core::Error::Msg(format!("tokenizer has no '{pad}' token")))?;
    let eos = m.tokenizer.token_to_id("<|endoftext|>");
    let mut ids: Vec<u32> = m
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        .get_ids()
        .to_vec();
    if ids.len() > max_len {
        ids.truncate(max_len);
        if let (Some(last), Some(e)) = (ids.last_mut(), eos) {
            *last = e; // keep the end-of-text token a CLIP model expects
        }
    }
    while ids.len() < max_len {
        ids.push(pad_id);
    }
    let tokens = Tensor::new(ids.as_slice(), &m.device)?.unsqueeze(0)?;
    candle_core::Module::forward(&m.clip, &tokens)
}

/// splitmix64 + Box-Muller: deterministic, identical on every device (Candle can only seed the CUDA RNG).
pub(crate) fn seeded_normal(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let unit = |x: u64| ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64; // (0, 1)
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (unit(next()), unit(next()));
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * u2).cos()) as f32);
        out.push((r * (2.0 * std::f64::consts::PI * u2).sin()) as f32);
    }
    out.truncate(n);
    out
}

fn fresh_seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    t ^ u64::from(std::process::id()).rotate_left(32)
}

#[cfg(test)]
pub(crate) fn encode_png_for_tests(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    encode_png(rgb, width, height).unwrap()
}

pub(crate) fn encode_png(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(rgb.len() / 2);
    {
        let mut enc = png::Encoder::new(&mut out, width, height);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(|e| AiKitError::Internal {
            reason: format!("png header: {e}"),
        })?;
        w.write_image_data(rgb).map_err(|e| AiKitError::Internal {
            reason: format!("png data: {e}"),
        })?;
    }
    Ok(out)
}

/// Build one model component from a safetensors file. The file is opened once, checked against the identity recorded when it was
/// hashed, and mapped from that same handle (Candle's own `build_*` helpers take a path and open it again, which would leave a gap for a
/// swap). The tensors are copied out by `build`, so the mapping is dropped when this returns; the handle is re-checked first, and a file
/// that changed during the build is reported in preference to whatever the build returned.
fn from_verified<T>(
    model_id: &str,
    path: &str,
    device: &Device,
    dtype: DType,
    what: &str,
    build: impl FnOnce(candle_nn::VarBuilder<'_>) -> candle_core::Result<T>,
) -> Result<T> {
    let fail = |e: candle_core::Error| AiKitError::ModelLoadFailed {
        model_id: model_id.to_string(),
        reason: format!("{what}: {e}"),
    };
    let maps = VerifiedMaps::open(&[path])?;
    let built = maps
        .var_builder(dtype, device)
        .and_then(build)
        .map_err(fail);
    maps.recheck()?;
    built
}

/// The SD-2.1 VAE configuration. `StableDiffusionConfig` keeps its component configs private and only offers path-taking builders, so
/// the values `StableDiffusionConfig::v2_1` uses are repeated here; a test compares them with what that constructor produces, so a
/// Candle upgrade that changes them fails loudly.
fn sd_vae_config() -> AutoEncoderKLConfig {
    AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

/// The SD-2.1 UNet configuration (see [`sd_vae_config`]).
fn sd_unet_config() -> UNet2DConditionModelConfig {
    let bc = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };
    UNet2DConditionModelConfig {
        blocks: vec![
            bc(320, Some(1), 5),
            bc(640, Some(1), 10),
            bc(1280, Some(1), 20),
            bc(1280, None, 20),
        ],
        center_input_sample: false,
        cross_attention_dim: 1024,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_commercial_only_service_refuses_a_named_non_commercial_model_before_loading_anything() {
        let d = tempfile::tempdir().unwrap();
        let cat = d.path().join("catalog.yaml");
        std::fs::write(
            &cat,
            r#"version: "1.0"
models:
  - id: "dev-model"
    version: "1.0.0"
    os_arch: ["x86_64"]
    quantization_variants: ["q4"]
    default_quantization: "q4"
    hash: "blake3-abcd1234"
    size_bytes: 1
    location: "file:///does-not-exist"
    metadata:
      parameters: 1
      context_window: 1
      vram_required_mb: 1
      inference_latency_ms: 1
      custom:
        kind: "image_generation"
        license: "CC-BY-NC-4.0"
"#,
        )
        .unwrap();
        let svc = CandleImageService::new(cat.to_str().unwrap(), DevicePreference::Cpu)
            .unwrap()
            .with_commercial_only();
        match svc.load_model("dev-model") {
            Err(AiKitError::ModelLoadFailed { reason, .. }) => {
                assert!(reason.contains("licence"), "{reason}")
            }
            other => panic!("expected a licence refusal, got {other:?}"),
        }
    }

    #[test]
    fn seeded_noise_is_deterministic_and_roughly_standard_normal() {
        let a = seeded_normal(7, 20_000);
        assert_eq!(a, seeded_normal(7, 20_000));
        assert_ne!(a, seeded_normal(8, 20_000));
        let n = a.len() as f32;
        let mean = a.iter().sum::<f32>() / n;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
        assert_eq!(seeded_normal(1, 5).len(), 5, "odd lengths are handled");
    }

    #[test]
    fn png_encoding_round_trips_the_header() {
        let rgb = vec![128u8; 16 * 8 * 3];
        let png = encode_png(&rgb, 16, 8).unwrap();
        assert_eq!(crate::modality::png_dims(&png), Some((16, 8)));
    }

    // ---- Stable Diffusion components are built from verified handles ----------------------------------------------------------------

    /// Write `vars` (what a VarMap-built module registered) as a verified safetensors file in `dir`.
    fn verified_weights(dir: &std::path::Path, name: &str, vm: &candle_nn::VarMap) -> String {
        let path = dir.join(name);
        vm.save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        // let the timestamp tick pass so a later in-place rewrite cannot share the stamp's tick
        std::thread::sleep(std::time::Duration::from_millis(50));
        let p = path.to_str().unwrap().to_string();
        let mut m = ModelMetadata {
            id: "sd".into(),
            version: "1".into(),
            os_arch: vec![],
            quantization_variants: vec![],
            default_quantization: String::new(),
            hash: blake3::hash(&bytes).to_hex().to_string(),
            size_bytes: bytes.len() as u64,
            location: format!("file://{p}"),
            metadata: crate::types::ModelMetadataExtended {
                parameters: 0,
                context_window: 0,
                vram_required_mb: 0,
                inference_latency_ms: 0,
                framework: String::new(),
                custom: HashMap::new(),
            },
            files: vec![],
        };
        m.id = name.into();
        crate::backends::common::verify_model_files(&m).unwrap();
        p
    }

    fn tiny_vae_cfg() -> AutoEncoderKLConfig {
        AutoEncoderKLConfig {
            block_out_channels: vec![32],
            layers_per_block: 1,
            ..AutoEncoderKLConfig::default()
        }
    }

    fn tiny_unet_cfg() -> UNet2DConditionModelConfig {
        let bc = |c, x, h| BlockConfig {
            out_channels: c,
            use_cross_attn: x,
            attention_head_dim: h,
        };
        UNet2DConditionModelConfig {
            blocks: vec![bc(32, Some(1), 2), bc(32, None, 2)],
            layers_per_block: 1,
            cross_attention_dim: 32,
            ..UNet2DConditionModelConfig::default()
        }
    }

    fn tiny_vae_file(dir: &std::path::Path) -> String {
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        AutoEncoderKL::new(vb, 3, 3, tiny_vae_cfg()).unwrap();
        verified_weights(dir, "vae.safetensors", &vm)
    }

    fn tiny_unet_file(dir: &std::path::Path) -> String {
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        UNet2DConditionModel::new(vb, 4, 4, false, tiny_unet_cfg()).unwrap();
        verified_weights(dir, "unet.safetensors", &vm)
    }

    fn build_vae(p: &str) -> Result<AutoEncoderKL> {
        from_verified("sd", p, &Device::Cpu, DType::F32, "vae", |vb| {
            AutoEncoderKL::new(vb, 3, 3, tiny_vae_cfg())
        })
    }

    fn build_unet(p: &str) -> Result<UNet2DConditionModel> {
        from_verified("sd", p, &Device::Cpu, DType::F32, "unet", |vb| {
            UNet2DConditionModel::new(vb, 4, 4, false, tiny_unet_cfg())
        })
    }

    #[test]
    fn the_sd_v21_component_configs_are_what_candle_builds_for_v2_1() {
        use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
        let shown = format!(
            "{:?}",
            StableDiffusionConfig::v2_1(None, Some(512), Some(512))
        );
        assert!(
            shown.contains(&format!("autoencoder: {:?}", sd_vae_config())),
            "{shown}"
        );
        assert!(
            shown.contains(&format!("unet: {:?}", sd_unet_config())),
            "{shown}"
        );
    }

    #[test]
    fn vae_and_unet_build_from_verified_files_and_refuse_a_swapped_one() {
        let d = tempfile::tempdir().unwrap();
        let (vae, unet) = (tiny_vae_file(d.path()), tiny_unet_file(d.path()));
        build_vae(&vae).unwrap();
        build_unet(&unet).unwrap();
        for (path, build) in [
            (
                &vae,
                (|p: &str| build_vae(p).map(drop)) as fn(&str) -> Result<()>,
            ),
            (&unet, |p: &str| build_unet(p).map(drop)),
        ] {
            let mut b = std::fs::read(path).unwrap();
            *b.last_mut().unwrap() ^= 0xff;
            std::fs::write(path, b).unwrap(); // same size, in place
            let e = build(path).unwrap_err();
            assert!(
                e.to_string().contains("changed since it was verified"),
                "{e}"
            );
        }
    }

    #[test]
    fn vae_and_unet_are_built_from_the_handle_that_was_checked_not_the_path() {
        for which in ["vae", "unet"] {
            let d = tempfile::tempdir().unwrap();
            let p = if which == "vae" {
                tiny_vae_file(d.path())
            } else {
                tiny_unet_file(d.path())
            };
            let _swap = crate::backends::common::seam::swap_once_after_check(&p);
            if which == "vae" {
                build_vae(&p).unwrap_or_else(|e| panic!("read by path: {e}"));
            } else {
                build_unet(&p).unwrap_or_else(|e| panic!("read by path: {e}"));
            }
        }
    }

    #[test]
    fn a_write_while_a_component_is_built_is_reported() {
        let d = tempfile::tempdir().unwrap();
        let p = tiny_vae_file(d.path());
        // re-check #1 is inside `VerifiedMaps::open`, #2 the one after the build
        let _w = crate::backends::common::seam::rewrite_on_recheck(&p, 2);
        let e = build_vae(&p).map(drop).expect_err("reported");
        assert!(
            e.to_string().contains("changed since it was verified"),
            "{e}"
        );
    }

    #[test]
    fn the_clip_builder_reads_the_checked_handle_too() {
        // the real CLIP config is far too large for a fixture, so this only shows which bytes are read: the original (valid
        // safetensors, missing the CLIP tensors) and not the garbage that replaced it
        let d = tempfile::tempdir().unwrap();
        let vm = candle_nn::VarMap::new();
        vm.get(
            (1,),
            "x",
            candle_nn::Init::Const(0.),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let p = verified_weights(d.path(), "clip.safetensors", &vm);
        let build = |p: &str| {
            from_verified("sd", p, &Device::Cpu, DType::F32, "text encoder", |vb| {
                ClipTextTransformer::new(vb, &ClipConfig::v2_1())
            })
            .map(drop)
            .expect_err("the tensors are not there")
            .to_string()
        };
        let _swap = crate::backends::common::seam::swap_once_after_check(&p);
        let e = build(&p);
        assert!(e.contains("cannot find tensor"), "{e}");
    }
}
