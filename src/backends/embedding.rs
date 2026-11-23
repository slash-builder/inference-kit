//! Text embeddings on Candle (a BERT-family sentence encoder such as all-MiniLM-L6-v2). **POC.**
//!
//! [`CandleEmbeddingService`] implements [`RequestService`] for [`RequestKind::Embedding`]: the
//! "semantic search over Courier files" use the Inference Kit design names. It mean-pools the last hidden state under the
//! attention mask and, if asked, scales each vector to unit length (so a dot product is cosine similarity).
//!
//! Embeddings of household files are about as private as data gets, so the router's default (`Private`: never leaves the
//! device) is the right one and this backend has no cloud counterpart. The catalog entry has `model.safetensors` as the main file and
//! `config` and `tokenizer` as extra files; every file is BLAKE3-verified on load.

use crate::backends::common::{
    check_vram, device_for, verify_model_files, DevicePreference, VerifiedFile, VerifiedMaps,
};
use crate::error::{AiKitError, Result};
use crate::modality::{
    unsupported, EmbeddingParams, EmbeddingResponse, Output, Request, RequestKind, RequestService,
};
use crate::registry::ModelRegistry;
use crate::types::ModelMetadata;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

struct LoadedEmbeddingModel {
    bert: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    version: String,
}

pub struct CandleEmbeddingService {
    registry: ModelRegistry,
    device_pref: DevicePreference,
    loaded: RwLock<HashMap<String, Arc<Mutex<LoadedEmbeddingModel>>>>,
}

impl CandleEmbeddingService {
    pub fn new(catalog_path: &str, device: DevicePreference) -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::load_from_file(catalog_path)?,
            device_pref: device,
            loaded: RwLock::new(HashMap::new()),
        })
    }

    fn model_meta(&self, id: &str) -> Result<ModelMetadata> {
        let m = self
            .registry
            .get(id)
            .cloned()
            .ok_or_else(|| AiKitError::ModelNotFound {
                model_id: id.to_string(),
            })?;
        if m.kind() != RequestKind::Embedding {
            return Err(AiKitError::ModelLoadFailed {
                model_id: id.to_string(),
                reason: format!(
                    "'{id}' is a {} model, not an embedding model",
                    m.kind().as_str()
                ),
            });
        }
        Ok(m)
    }

    pub fn load_model(&self, id: &str) -> Result<()> {
        if self.loaded.read().is_ok_and(|l| l.contains_key(id)) {
            return Ok(());
        }
        let model = self.model_meta(id)?;
        let fail = |reason: String| AiKitError::ModelLoadFailed {
            model_id: id.to_string(),
            reason,
        };
        let (device, _) = device_for(self.device_pref).map_err(fail)?;
        check_vram(&device, &model)?;
        let weights = verify_model_files(&model)?;
        let need = |role: &str| {
            model
                .file_path(role)
                .ok_or_else(|| fail(format!("catalog entry has no '{role}' file")))
        };
        let (cfg_path, tok_path) = (need("config")?, need("tokenizer")?);
        // read through handles checked against the hashes just taken, not re-opened by path
        let config: Config = serde_json::from_slice(&VerifiedFile::open(&cfg_path)?.read_all()?)
            .map_err(|e| fail(format!("config: {e}")))?;
        let mut tokenizer = Tokenizer::from_bytes(VerifiedFile::open(&tok_path)?.read_all()?)
            .map_err(|e| fail(format!("tokenizer: {e}")))?;
        tokenizer.with_padding(Some(PaddingParams::default()));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_position_embeddings.min(256),
                ..Default::default()
            }))
            .map_err(|e| fail(format!("tokenizer: {e}")))?;
        // Mapped from a handle checked against the hash `verify_model_files` just took (see `VerifiedMaps`), not re-opened by path. The
        // tensors are copied out by `BertModel::load`, so the mapping is dropped when this function returns.
        let maps = VerifiedMaps::open(&[weights.as_str()])?;
        let vb = maps
            .var_builder(DType::F32, &device)
            .map_err(|e| fail(format!("weights: {e}")))?;
        let bert = BertModel::load(vb, &config).map_err(|e| fail(format!("model: {e}")));
        // a write while the weights were being read is reported in preference to whatever the load returned
        maps.recheck()?;
        let bert = bert?;
        self.loaded
            .write()
            .map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".into(),
            })?
            .insert(
                id.to_string(),
                Arc::new(Mutex::new(LoadedEmbeddingModel {
                    bert,
                    tokenizer,
                    device,
                    version: model.version.clone(),
                })),
            );
        Ok(())
    }

    fn embed(&self, p: &EmbeddingParams) -> Result<EmbeddingResponse> {
        Request::Embedding(p.clone()).validate()?;
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
        let m = model.lock().map_err(|_| AiKitError::Internal {
            reason: "embedding model lock poisoned".into(),
        })?;
        let started = std::time::Instant::now();
        let internal = |what: &str, e: candle_core::Error| AiKitError::Internal {
            reason: format!("{what}: {e}"),
        };

        let enc = m
            .tokenizer
            .encode_batch(p.inputs.clone(), true)
            .map_err(|e| AiKitError::Internal {
                reason: format!("tokenize: {e}"),
            })?;
        let prompt_tokens: u32 = enc
            .iter()
            .map(|e| e.get_attention_mask().iter().sum::<u32>())
            .sum();
        let rows = |f: &dyn Fn(&tokenizers::Encoding) -> Vec<u32>| -> candle_core::Result<Tensor> {
            let v: Vec<Tensor> = enc
                .iter()
                .map(|e| Tensor::new(f(e).as_slice(), &m.device))
                .collect::<candle_core::Result<_>>()?;
            Tensor::stack(&v, 0)
        };
        let vectors = (|| -> candle_core::Result<Vec<Vec<f32>>> {
            let ids = rows(&|e| e.get_ids().to_vec())?;
            let mask = rows(&|e| e.get_attention_mask().to_vec())?;
            let types = ids.zeros_like()?;
            let hidden = m.bert.forward(&ids, &types, Some(&mask))?; // (batch, seq, dim)
            let maskf = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
            let summed = hidden.broadcast_mul(&maskf)?.sum(1)?;
            let mut pooled = summed.broadcast_div(&maskf.sum(1)?)?;
            if p.normalize {
                pooled = pooled.broadcast_div(&pooled.sqr()?.sum_keepdim(1)?.sqrt()?)?;
            }
            pooled.to_vec2::<f32>()
        })()
        .map_err(|e| internal("embed", e))?;

        let dimensions = vectors.first().map_or(0, Vec::len);
        Ok(EmbeddingResponse {
            vectors,
            dimensions,
            prompt_tokens,
            inference_ms: u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX),
            model_version: m.version.clone(),
            session_id: p.session_id.clone().unwrap_or_default(),
            request_id: p
                .request_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        })
    }
}

#[async_trait]
impl RequestService for CandleEmbeddingService {
    fn kinds(&self) -> Vec<RequestKind> {
        vec![RequestKind::Embedding]
    }

    async fn handle(&self, req: Request) -> Result<Output> {
        match req {
            Request::Embedding(p) => self.embed(&p).map(Output::Embeddings),
            other => Err(unsupported(other.kind(), "this is an embedding backend")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};
    use tokenizers::models::wordlevel::WordLevel;

    /// A catalog entry for a tiny random-free BERT written to disk: weights, config and tokenizer, all hashed.
    fn tiny_catalog(dir: &std::path::Path, tamper_weights_hash: bool) -> String {
        let config = Config {
            vocab_size: 8,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            ..Config::default()
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        BertModel::load(vb, &config).unwrap();
        let weights = dir.join("model.safetensors");
        vm.save(&weights).unwrap();
        let cfg = dir.join("config.json");
        std::fs::write(
            &cfg,
            r#"{"vocab_size":8,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":16,
            "hidden_act":"gelu","hidden_dropout_prob":0.0,"max_position_embeddings":16,"type_vocab_size":2,
            "initializer_range":0.02,"layer_norm_eps":1e-12,"pad_token_id":1}"#,
        )
        .unwrap();
        let tok = dir.join("tokenizer.json");
        let vocab = ["[UNK]", "[PAD]", "hello", "world"]
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut t = Tokenizer::new(model);
        t.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace {}));
        t.save(&tok, false).unwrap();

        let h = |p: &std::path::Path| {
            blake3::hash(&std::fs::read(p).unwrap())
                .to_hex()
                .to_string()
        };
        let size = |p: &std::path::Path| std::fs::metadata(p).unwrap().len();
        let w_hash = if tamper_weights_hash {
            "0".repeat(64)
        } else {
            h(&weights)
        };
        let catalog = format!(
            r#"
version: "1.0"
models:
  - id: "tiny"
    version: "1.0.0"
    os_arch: ["x86_64"]
    quantization_variants: ["f32"]
    default_quantization: "f32"
    hash: "{w_hash}"
    size_bytes: {}
    location: "file://{}"
    metadata:
      parameters: 1
      context_window: 16
      vram_required_mb: 1
      inference_latency_ms: 1
      custom:
        kind: "embedding"
    files:
      - role: "config"
        location: "file://{}"
        hash: "{}"
        size_bytes: {}
      - role: "tokenizer"
        location: "file://{}"
        hash: "{}"
        size_bytes: {}
"#,
            size(&weights),
            weights.display(),
            cfg.display(),
            h(&cfg),
            size(&cfg),
            tok.display(),
            h(&tok),
            size(&tok),
        );
        let path = dir.join("catalog.yaml");
        std::fs::write(&path, catalog).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn a_verified_model_loads_through_the_checked_handles_and_embeds() {
        let d = tempfile::tempdir().unwrap();
        let svc =
            CandleEmbeddingService::new(&tiny_catalog(d.path(), false), DevicePreference::Cpu)
                .unwrap();
        svc.load_model("tiny").unwrap();
        let out = svc
            .embed(&EmbeddingParams {
                model_id: "tiny".into(),
                inputs: vec!["hello world".into()],
                normalize: false,
                session_id: None,
                request_id: None,
            })
            .unwrap();
        assert_eq!(out.dimensions, 8);
    }

    #[test]
    fn weights_config_and_tokenizer_are_each_read_from_the_handle_that_was_checked() {
        // after a file's identity check, a different file takes its path; the load must still use the original bytes
        for name in ["model.safetensors", "config.json", "tokenizer.json"] {
            let d = tempfile::tempdir().unwrap();
            let svc =
                CandleEmbeddingService::new(&tiny_catalog(d.path(), false), DevicePreference::Cpu)
                    .unwrap();
            let target = d.path().join(name).to_str().unwrap().to_string();
            let _swap = crate::backends::common::seam::swap_once_after_check(&target);
            svc.load_model("tiny")
                .unwrap_or_else(|e| panic!("{name} was read by path: {e}"));
        }
    }

    #[test]
    fn a_write_while_the_weights_are_being_read_fails_the_load() {
        let d = tempfile::tempdir().unwrap();
        let svc =
            CandleEmbeddingService::new(&tiny_catalog(d.path(), false), DevicePreference::Cpu)
                .unwrap();
        let w = d.path().join("model.safetensors");
        // re-check #1 is inside `VerifiedMaps::open`; #2 is the one after the load
        let _w = crate::backends::common::seam::rewrite_on_recheck(w.to_str().unwrap(), 2);
        let e = svc.load_model("tiny").unwrap_err();
        assert!(
            e.to_string().contains("changed since it was verified"),
            "{e}"
        );
        assert!(!svc.loaded.read().unwrap().contains_key("tiny"));
    }

    #[test]
    fn a_model_whose_weights_do_not_match_the_catalog_never_loads() {
        let d = tempfile::tempdir().unwrap();
        let svc = CandleEmbeddingService::new(&tiny_catalog(d.path(), true), DevicePreference::Cpu)
            .unwrap();
        assert!(matches!(
            svc.load_model("tiny"),
            Err(AiKitError::IntegrityCheckFailed { .. })
        ));
        assert!(!svc.loaded.read().unwrap().contains_key("tiny"));
    }
}
