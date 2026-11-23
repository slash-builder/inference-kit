//! Model registry and discovery.
//!
//! # Design Notes
//!
//! The model registry is loaded from `/usr/share/models/catalog.yaml` (OS-shipped, immutable).
//! It holds metadata for all available models, including:
//! - Semantic version
//! - BLAKE3 hash
//! - Quantization variants
//! - VRAM requirements
//! - OS architecture support
//!
//! # Invariant
//!
//! Models are OS resources (decision invariant #1). They ship signed with the OS image
//! and are verified at boot. Registry is immutable post-deployment.
//!
//! # TODO
//!
//! - Implement catalog.yaml loader (YAML parsing).
//! - Integration with Storage Kit once available.
//! - Versioning semantics (SemVer resolver for app dependencies).

use crate::error::Result;
use crate::types::ModelMetadata;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// On-disk shape of `catalog.yaml`, matching the doc-commented example on
/// [`ModelRegistry::load_from_file`] exactly.
#[derive(Debug, Deserialize)]
struct CatalogFile {
    version: String,
    models: Vec<ModelMetadata>,
}

/// Model registry (in-memory catalog, loaded from OS image).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    /// Models indexed by ID.
    pub models: HashMap<String, ModelMetadata>,

    /// Catalog source (e.g., "file:///usr/share/models/catalog.yaml").
    pub source: String,

    /// Timestamp when catalog was loaded.
    pub loaded_at: u64,

    /// Checksum or version of the catalog (for audit trail).
    pub catalog_version: String,
}

impl ModelRegistry {
    /// Create a new empty registry.
    pub fn new(source: String, catalog_version: String) -> Self {
        ModelRegistry {
            models: HashMap::new(),
            source,
            loaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            catalog_version,
        }
    }

    /// Load registry from YAML file.
    ///
    /// # Examples
    ///
    /// ```yaml
    /// # /usr/share/models/catalog.yaml
    /// version: "1.0"
    /// models:
    ///   - id: "llama2-7b"
    ///     version: "2.0.0"
    ///     os_arch: ["x86_64", "aarch64"]
    ///     quantization_variants: ["q4", "q8", "fp16"]
    ///     default_quantization: "q4"
    ///     hash: "blake3-hash..."
    ///     size_bytes: 3865470976
    ///     location: "file:///usr/share/models/llama2-7b-q4.bin"
    ///     metadata:
    ///       parameters: 7000000000
    ///       context_window: 4096
    ///       vram_required_mb: 4096
    ///       inference_latency_ms: 45
    /// ```
    ///
    /// # Errors
    ///
    /// - `AiKitError::Io` if the file can't be read.
    /// - `AiKitError::YamlError` if the file doesn't parse to the shape above.
    /// - Whatever [`ModelRegistry::validate`] returns, if any parsed model fails validation.
    pub fn load_from_file(path: &str) -> Result<ModelRegistry> {
        let contents = std::fs::read_to_string(path)?;
        let catalog: CatalogFile = crate::yaml::from_str(&contents)?;

        let mut registry = ModelRegistry::new(format!("file://{path}"), catalog.version);
        for model in catalog.models {
            registry.add(model);
        }
        registry.validate()?;
        Ok(registry)
    }

    /// Look up a model by ID.
    pub fn get(&self, model_id: &str) -> Option<&ModelMetadata> {
        self.models.get(model_id)
    }

    /// Check if a model exists.
    pub fn has(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    /// List all available models.
    pub fn list(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }

    /// The model to use for `kind` when the caller names none.
    ///
    /// An entry marked `custom.default: "true"` wins; otherwise the first by id (the registry is unordered, so this is made
    /// deterministic). A model that is not known to permit commercial use (a non-commercial or research licence **or** one that is not
    /// stated or not recognised, see [`ModelMetadata::licence_class`]) is **never** chosen unless `allow_noncommercial` is set, so a
    /// development-only model such as `sd-turbo` cannot silently become what a product ships.
    ///
    /// This covers the default only. A caller that names a model id bypasses it: use [`ModelRegistry::check_licence`] for that.
    pub fn default_for(
        &self,
        kind: crate::modality::RequestKind,
        allow_noncommercial: bool,
    ) -> Option<&ModelMetadata> {
        let mut ok: Vec<&ModelMetadata> = self
            .models
            .values()
            .filter(|m| m.kind() == kind)
            .filter(|m| {
                allow_noncommercial || m.licence_class() == crate::types::LicenceClass::Commercial
            })
            .collect();
        ok.sort_by(|a, b| a.id.cmp(&b.id));
        ok.iter()
            .find(|m| {
                m.metadata
                    .custom
                    .get("default")
                    .is_some_and(|d| d == "true")
            })
            .or(ok.first())
            .copied()
    }

    /// Refuse a model that is not known to permit commercial use, unless `allow_noncommercial`. For callers that name a model id, which
    /// [`ModelRegistry::default_for`] never sees. An id that is not in the registry passes: reporting it missing is the service's job.
    pub fn check_licence(&self, model_id: &str, allow_noncommercial: bool) -> Result<()> {
        match self.models.get(model_id) {
            Some(m)
                if !allow_noncommercial
                    && m.licence_class() != crate::types::LicenceClass::Commercial =>
            {
                Err(crate::error::AiKitError::ModelLoadFailed {
                    model_id: model_id.to_string(),
                    reason: format!(
                        "licence is {:?} (not known to permit commercial use); \
                         set `commercial_use: \"true\"` in the catalog entry if that is wrong, or opt in to non-commercial models",
                        m.licence_class()
                    ),
                })
            }
            _ => Ok(()),
        }
    }

    /// Add a model to the registry (for testing or dynamic registration).
    pub fn add(&mut self, model: ModelMetadata) {
        self.models.insert(model.id.clone(), model);
    }

    /// Get number of models in registry.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Validate that all models are properly configured.
    ///
    /// # Checks
    ///
    /// - All model IDs are non-empty.
    /// - All hashes are valid (BLAKE3).
    /// - At least one quantization variant per model.
    /// - Default quantization is in variants list.
    /// - VRAM requirements are positive.
    pub fn validate(&self) -> Result<()> {
        for model in self.models.values() {
            if model.id.is_empty() {
                return Err(crate::error::AiKitError::Internal {
                    reason: "Model ID cannot be empty".to_string(),
                });
            }

            if model.hash.is_empty() {
                return Err(crate::error::AiKitError::Internal {
                    reason: format!("Model {} has empty hash", model.id),
                });
            }

            if model.quantization_variants.is_empty() {
                return Err(crate::error::AiKitError::Internal {
                    reason: format!("Model {} has no quantization variants", model.id),
                });
            }

            if !model
                .quantization_variants
                .contains(&model.default_quantization)
            {
                return Err(crate::error::AiKitError::Internal {
                    reason: format!(
                        "Model {} default quantization {} not in variants",
                        model.id, model.default_quantization
                    ),
                });
            }

            if let Some(c) = model.metadata.custom.get("kind") {
                if c.parse::<crate::modality::RequestKind>().is_err() {
                    return Err(crate::error::AiKitError::Internal {
                        reason: format!("Model {} has unknown request kind '{c}'", model.id),
                    });
                }
            }
            if let Some(f) = model
                .files
                .iter()
                .find(|f| f.hash.is_empty() || f.role.is_empty())
            {
                return Err(crate::error::AiKitError::Internal {
                    reason: format!(
                        "Model {} has a file ('{}') with no role or no hash",
                        model.id, f.role
                    ),
                });
            }

            if model.metadata.vram_required_mb == 0 {
                return Err(crate::error::AiKitError::Internal {
                    reason: format!("Model {} has zero VRAM requirement", model.id),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelMetadataExtended;

    fn create_test_model() -> ModelMetadata {
        ModelMetadata {
            id: "test-model".to_string(),
            version: "1.0.0".to_string(),
            os_arch: vec!["x86_64".to_string()],
            quantization_variants: vec!["q4".to_string(), "q8".to_string()],
            default_quantization: "q4".to_string(),
            hash: "blake3-abcd1234".to_string(),
            size_bytes: 1000000,
            location: "file:///usr/share/models/test-model-q4.bin".to_string(),
            metadata: ModelMetadataExtended {
                parameters: 1000000,
                context_window: 2048,
                vram_required_mb: 2048,
                inference_latency_ms: 50,
                framework: "candle".to_string(),
                custom: Default::default(),
            },
            files: Vec::new(),
        }
    }

    #[test]
    fn test_registry_create() {
        let reg = ModelRegistry::new(
            "file:///usr/share/models/catalog.yaml".to_string(),
            "1.0.0".to_string(),
        );
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_add_and_get() {
        let mut reg = ModelRegistry::new(
            "file:///usr/share/models/catalog.yaml".to_string(),
            "1.0.0".to_string(),
        );
        let model = create_test_model();
        reg.add(model.clone());

        assert_eq!(reg.len(), 1);
        assert!(reg.has("test-model"));
        assert_eq!(reg.get("test-model").unwrap().id, "test-model");
    }

    #[test]
    fn test_registry_validation_valid() {
        let mut reg = ModelRegistry::new(
            "file:///usr/share/models/catalog.yaml".to_string(),
            "1.0.0".to_string(),
        );
        let model = create_test_model();
        reg.add(model);

        assert!(reg.validate().is_ok());
    }

    #[test]
    fn test_registry_validation_invalid_quantization() {
        let mut reg = ModelRegistry::new(
            "file:///usr/share/models/catalog.yaml".to_string(),
            "1.0.0".to_string(),
        );
        let mut model = create_test_model();
        model.default_quantization = "q16".to_string(); // Not in variants

        reg.add(model);
        assert!(reg.validate().is_err());
    }

    #[test]
    fn test_load_from_file_valid_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.yaml");
        std::fs::write(
            &path,
            r#"
version: "1.0"
models:
  - id: "test-model"
    version: "1.0.0"
    os_arch: ["x86_64", "aarch64"]
    quantization_variants: ["q4", "q8"]
    default_quantization: "q4"
    hash: "blake3-abcd1234"
    size_bytes: 1000000
    location: "file:///dev-models/test-model/model.gguf"
    metadata:
      parameters: 1000000
      context_window: 2048
      vram_required_mb: 2048
      inference_latency_ms: 50
"#,
        )
        .unwrap();

        let reg = ModelRegistry::load_from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(reg.catalog_version, "1.0");
        assert_eq!(reg.len(), 1);
        assert!(reg.has("test-model"));
        assert_eq!(reg.get("test-model").unwrap().hash, "blake3-abcd1234");
    }

    #[test]
    fn test_load_from_file_missing_file() {
        let result = ModelRegistry::load_from_file("/nonexistent/catalog.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_file_malformed_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.yaml");
        std::fs::write(&path, "this: [is, not: valid: yaml").unwrap();

        let result = ModelRegistry::load_from_file(path.to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_file_invalid_model_fails_validation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.yaml");
        // default_quantization "q16" is not in quantization_variants -> validate() should fail.
        std::fs::write(
            &path,
            r#"
version: "1.0"
models:
  - id: "test-model"
    version: "1.0.0"
    os_arch: ["x86_64"]
    quantization_variants: ["q4"]
    default_quantization: "q16"
    hash: "blake3-abcd1234"
    size_bytes: 1000000
    location: "file:///dev-models/test-model/model.gguf"
    metadata:
      parameters: 1000000
      context_window: 2048
      vram_required_mb: 2048
      inference_latency_ms: 50
"#,
        )
        .unwrap();

        let result = ModelRegistry::load_from_file(path.to_str().unwrap());
        assert!(result.is_err());
    }

    fn img(id: &str, licence: Option<&str>, default: bool) -> ModelMetadata {
        let mut m = create_test_model();
        m.id = id.to_string();
        m.metadata
            .custom
            .insert("kind".into(), "image_generation".into());
        if let Some(l) = licence {
            m.metadata.custom.insert("license".into(), l.into());
        }
        if default {
            m.metadata.custom.insert("default".into(), "true".into());
        }
        m
    }

    #[test]
    fn a_non_commercial_model_is_never_the_default_unless_asked_for() {
        let mut r = ModelRegistry::new("t".into(), "1".into());
        r.add(img("a-sd-turbo", Some("non-commercial research"), true)); // sorts first AND is flagged default
        r.add(img("z-image-turbo", Some("Apache-2.0"), false));
        let k = crate::modality::RequestKind::ImageGeneration;
        assert_eq!(r.default_for(k, false).unwrap().id, "z-image-turbo");
        assert_eq!(
            r.default_for(k, true).unwrap().id,
            "a-sd-turbo",
            "opting in allows it"
        );
    }

    #[test]
    fn a_flagged_default_beats_alphabetical_order_and_no_candidate_is_none() {
        let mut r = ModelRegistry::new("t".into(), "1".into());
        let k = crate::modality::RequestKind::ImageGeneration;
        assert!(r.default_for(k, false).is_none());
        r.add(img("a", Some("MIT"), false));
        r.add(img("b", Some("Apache-2.0"), true));
        assert_eq!(r.default_for(k, false).unwrap().id, "b");
        assert!(r
            .default_for(crate::modality::RequestKind::Embedding, false)
            .is_none());
    }

    fn with_licence(licence: Option<&str>) -> ModelMetadata {
        img("m", licence, false)
    }

    #[test]
    fn common_non_commercial_licence_names_are_non_commercial() {
        use crate::types::LicenceClass::NonCommercial;
        for l in [
            "non-commercial research",
            "Non-Commercial",
            "noncommercial",
            "research-only",
            "CC-BY-NC-4.0",
            "cc-by-nc-sa-4.0",
            "CC BY-NC-ND 4.0",
            "cc_by_nc_4.0",
            "FLUX.1-dev Non-Commercial License",
            "Stability AI Non Commercial Research Community License",
            "Academic use only",
            "evaluation only",
        ] {
            assert_eq!(with_licence(Some(l)).licence_class(), NonCommercial, "{l}");
        }
    }

    #[test]
    fn known_permissive_licences_are_commercial_and_anything_else_is_unknown() {
        use crate::types::LicenceClass::{Commercial, Unknown};
        for l in [
            "Apache-2.0",
            "apache 2.0",
            "MIT",
            "BSD-3-Clause",
            "ISC",
            "MIT License",
            "Apache License 2.0",
            "BSD 3-Clause License",
            "Apache-2.0 OR MIT",
            "CC-BY-3.0",
            "Apache-2.0-or-later",
            "CC0-1.0",
            "Unlicense",
        ] {
            assert_eq!(with_licence(Some(l)).licence_class(), Commercial, "{l}");
        }
        // not stated, or a name nobody here has classified: not assumed to be commercial
        for l in [
            None,
            Some(""),
            Some("  "),
            Some("other"),
            Some("proprietary"),
            Some("custom licence v3"),
            Some("see LICENSE file"),
            // deliberately not recognised as permissive: copyleft, RAIL and community licences are refused by default and a
            // catalog author vouches for them with `commercial_use: "true"`
            Some("GPL-3.0"),
            Some("LGPL-2.1"),
            Some("MPL-2.0"),
            Some("openrail++"),
            Some("CreativeML OpenRAIL-M"),
        ] {
            assert_eq!(with_licence(l).licence_class(), Unknown, "{l:?}");
        }
        // "nc" is matched as a word of the name, not as letters inside another word
        assert_ne!(
            with_licence(Some("Apache-2.0 (Lancaster fork)")).licence_class(),
            crate::types::LicenceClass::NonCommercial
        );
    }

    #[test]
    fn a_release_descriptor_does_not_make_a_permissive_licence_non_commercial() {
        use crate::types::LicenceClass::{Commercial, NonCommercial};
        assert_eq!(
            with_licence(Some("Apache-2.0 (research preview)")).licence_class(),
            Commercial
        );
        assert_eq!(
            with_licence(Some("MIT personal fork")).licence_class(),
            Commercial
        );
        assert_eq!(
            with_licence(Some("MIT research")).licence_class(),
            NonCommercial
        );
    }

    #[test]
    fn every_non_commercial_phrasing_fails_closed_even_with_commercial_use_true() {
        use crate::types::LicenceClass::NonCommercial;
        for l in [
            "Not for commercial use",
            "Commercial use prohibited",
            "Free for non-profit use",
            "Nicht kommerziell",
            "No commercial use",
            "Not commercial",
            "non commercial",
            "Research use only",
            "CC-BY-NC-4.0",
            "Apache-2.0 for personal use",
            "don't use commercially",
            "cannot be used for commercial purposes",
            "Commercial use unauthorized",
            "Commercial use needs permission",
            "Uso comercial prohibido",
            "Usage commercial interdit",
            "Nutzung kommerziell verboten",
            "Vietato uso commerciale",
            "MIT; commercial=false",
            "商用禁止",
            "禁止商用",
            "非商用",
            "MIT evaluation version",
            "MIT personal edition",
            "MIT (research model)",
            "not for commercial use, no restrictions on research",
        ] {
            let mut m = with_licence(Some(l));
            assert_eq!(m.licence_class(), NonCommercial, "{l}");
            m.metadata
                .custom
                .insert("commercial_use".into(), "true".into());
            assert_eq!(
                m.licence_class(),
                NonCommercial,
                "{l}: a contradicting commercial_use must not open it"
            );
        }
    }

    #[test]
    fn an_unrecognised_licence_is_refused_until_the_catalog_vouches_for_it() {
        use crate::types::LicenceClass::{Commercial, Unknown};
        for l in ["GPL-3.0", "openrail++", "CreativeML OpenRAIL-M"] {
            let mut m = with_licence(Some(l));
            assert_eq!(m.licence_class(), Unknown, "{l}");
            m.metadata
                .custom
                .insert("commercial_use".into(), "true".into());
            assert_eq!(m.licence_class(), Commercial, "{l} once vouched for");
        }
    }

    #[test]
    fn an_explicit_commercial_use_field_decides_but_a_non_commercial_name_still_wins() {
        use crate::types::LicenceClass::{Commercial, NonCommercial};
        let mut m = with_licence(Some("custom licence v3"));
        m.metadata
            .custom
            .insert("commercial_use".into(), "true".into());
        assert_eq!(
            m.licence_class(),
            Commercial,
            "the catalog author vouches for it"
        );
        m.metadata
            .custom
            .insert("commercial_use".into(), "false".into());
        assert_eq!(m.licence_class(), NonCommercial);
        let mut nc = with_licence(Some("CC-BY-NC-4.0"));
        nc.metadata
            .custom
            .insert("commercial_use".into(), "true".into());
        assert_eq!(
            nc.licence_class(),
            NonCommercial,
            "a contradiction fails closed"
        );
    }

    #[test]
    fn a_model_with_no_stated_licence_is_never_the_default_unless_asked_for() {
        let mut r = ModelRegistry::new("t".into(), "1".into());
        r.add(img("a-unstated", None, true)); // sorts first and is flagged default
        r.add(img("b-mit", Some("MIT"), false));
        let k = crate::modality::RequestKind::ImageGeneration;
        assert_eq!(r.default_for(k, false).unwrap().id, "b-mit");
        assert_eq!(r.default_for(k, true).unwrap().id, "a-unstated");
        let mut only_unknown = ModelRegistry::new("t".into(), "1".into());
        only_unknown.add(img("x", None, false));
        assert!(only_unknown.default_for(k, false).is_none());
    }

    #[test]
    fn naming_a_model_explicitly_is_checked_against_its_licence() {
        let mut r = ModelRegistry::new("t".into(), "1".into());
        r.add(img("nc", Some("CC-BY-NC-4.0"), false));
        r.add(img("unk", None, false));
        r.add(img("ok", Some("Apache-2.0"), false));
        assert!(r.check_licence("ok", false).is_ok());
        assert!(r.check_licence("nc", false).is_err());
        assert!(r.check_licence("unk", false).is_err());
        assert!(r.check_licence("nc", true).is_ok(), "opting in allows it");
        assert!(
            r.check_licence("not-in-the-catalog", false).is_ok(),
            "an unknown id is the service's ModelNotFound to report, not a licence matter"
        );
    }

    fn with_field(licence: &str, field: Option<&str>) -> ModelMetadata {
        let mut m = with_licence(Some(licence));
        if let Some(v) = field {
            m.metadata.custom.insert("commercial_use".into(), v.into());
        }
        m
    }

    #[test]
    fn the_commercial_use_field_is_trimmed_and_case_insensitive() {
        use crate::types::LicenceClass::{Commercial, NonCommercial, Unknown};
        for v in [
            "true", "True", "TRUE", " TRUE ", "yes", "Yes", "YES", "\tyes\n",
        ] {
            assert_eq!(
                with_field("custom", Some(v)).licence_class(),
                Commercial,
                "{v:?}"
            );
        }
        for v in ["false", "False", " FALSE ", "no", "No", " NO "] {
            assert_eq!(
                with_field("custom", Some(v)).licence_class(),
                NonCommercial,
                "{v:?}"
            );
            // and it wins over a permissive name
            assert_eq!(
                with_field("MIT", Some(v)).licence_class(),
                NonCommercial,
                "{v:?}"
            );
        }
        // anything else is not an answer: the name decides
        for v in ["", "maybe", "1", "ok"] {
            assert_eq!(
                with_field("custom", Some(v)).licence_class(),
                Unknown,
                "{v:?}"
            );
            assert_eq!(
                with_field("MIT", Some(v)).licence_class(),
                Commercial,
                "{v:?}"
            );
        }
    }

    #[test]
    fn affirmative_wording_is_commercial_with_and_without_the_field() {
        use crate::types::LicenceClass::{Commercial, NonCommercial};
        for l in [
            "MIT, no commercial restrictions",
            "Apache 2.0 - commercial use permitted, no restrictions",
            "Apache-2.0 (commercial use OK, no attribution required)",
            "MIT-0",
            "Licensed under MIT",
        ] {
            assert_eq!(with_field(l, None).licence_class(), Commercial, "{l}");
            assert_eq!(
                with_field(l, Some("false")).licence_class(),
                NonCommercial,
                "{l}"
            );
        }
    }
}
