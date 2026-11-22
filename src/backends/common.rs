//! Helpers shared by every Candle-backed service (text, image, embedding, and future kinds): where a model runs, whether
//! it fits, and whether its files are the ones the catalog promised.
//!
//! These used to live in `candle.rs` and were reached into by the other backends. Moving them here means a new backend
//! (for example Z-Image) depends on this module, not on the text backend.

use crate::error::{AiKitError, Result};
use crate::types::ModelMetadata;
use candle_core::safetensors::SliceSafetensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use std::fs::File;
use std::io::Read;
use std::str::FromStr;

/// Where a model runs. The default for [`CandleInferenceService::new`] is
/// [`DevicePreference::Cpu`], so nothing changes unless a caller opts in.
///
/// # No silent fallback
///
/// `Cuda(n)` means exactly that device: if it is not there, or the model does
/// not fit in its free memory, `load_model` **fails** (see invariant "fail
/// hard on OOM / no degrade"). Only `Auto` chooses for you, and it picks CUDA
/// device 0 when one is usable and the CPU otherwise; the choice is visible
/// afterwards through [`CandleInferenceService::device_of`], never hidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    /// Run on the CPU (the default).
    #[default]
    Cpu,
    /// Run on this CUDA device ordinal. Requires the `cuda` feature at build time.
    Cuda(usize),
    /// CUDA device 0 if usable, otherwise CPU.
    Auto,
}

impl FromStr for DevicePreference {
    type Err = String;
    /// Accepts `cpu`, `auto`, `cuda` (= `cuda:0`) and `cuda:N`.
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda(0)),
            other => other
                .strip_prefix("cuda:")
                .and_then(|n| n.parse::<usize>().ok())
                .map(Self::Cuda)
                .ok_or_else(|| {
                    format!("unknown device '{s}' (expected cpu, auto, cuda or cuda:N)")
                }),
        }
    }
}

/// Headroom added to the weight size when checking that a model fits in free
/// GPU memory: KV cache, activations and the CUDA context itself.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const VRAM_OVERHEAD_FRACTION: f64 = 0.15;
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const VRAM_OVERHEAD_FLOOR_MB: u64 = 384;

/// Turn a [`DevicePreference`] into a device and a label, failing hard (no silent CPU fallback) unless `Auto`.
pub(crate) fn device_for(pref: DevicePreference) -> std::result::Result<(Device, String), String> {
    match pref {
        DevicePreference::Cpu => Ok((Device::Cpu, "cpu".to_string())),
        DevicePreference::Cuda(n) => cuda_device(n),
        DevicePreference::Auto => Ok(cuda_device(0).unwrap_or((Device::Cpu, "cpu".to_string()))),
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Verified model files
//
// A model file is hashed against the catalog once, at load. Everything after that (image weights are re-opened on every request)
// must use **the file that was hashed**, not whatever the path names by then. So nothing in this module checks a path and then
// opens it: [`VerifiedFile::open`] opens the path once and checks the open handle (`fstat`) against what was recorded when that
// file was hashed, and every later read or mapping goes through that same handle. Renaming a different file over the path between
// the check and the read therefore cannot change what is read.
// ---------------------------------------------------------------------------------------------------------------------------------

/// Unix identity of a file, read from an open handle. `ctime` is the one that matters: only the kernel sets it, to the time of the last
/// write or metadata change, so restoring an old modification time (`utimensat`) after a rewrite still moves it.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UnixIdentity {
    dev: u64,
    ino: u64,
    mtime: i64,
    mtime_nsec: i64,
    ctime: i64,
    ctime_nsec: i64,
}

/// What identifies a file on disk without reading it, taken from an open handle: size and modification time everywhere; on Unix also
/// device, inode and change time.
///
/// What this cannot see: a same-size rewrite in place by a principal who can write the file, made within one timestamp tick of the
/// moment the file was hashed (coarse-timestamp filesystems tick in milliseconds to seconds), or on a filesystem that does not
/// maintain `ctime`. Hashing is the only way to see those, and re-hashing gigabytes per request is not affordable, so the remaining
/// assumption is that nothing with write access to the model directory modifies a model file while the process runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    #[cfg(unix)]
    unix: UnixIdentity,
    #[cfg(not(unix))]
    mtime: Option<std::time::SystemTime>,
}

/// `fstat` of an open file (never of a path).
fn stamp_of(file: &File) -> std::io::Result<FileStamp> {
    let m = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(FileStamp {
            len: m.len(),
            unix: UnixIdentity {
                dev: m.dev(),
                ino: m.ino(),
                mtime: m.mtime(),
                mtime_nsec: m.mtime_nsec(),
                ctime: m.ctime(),
                ctime_nsec: m.ctime_nsec(),
            },
        })
    }
    #[cfg(not(unix))]
    {
        Ok(FileStamp {
            len: m.len(),
            mtime: m.modified().ok(),
        })
    }
}

/// The files this process has hashed against the catalog, with the stamp the hashed handle had.
fn verified() -> &'static std::sync::Mutex<std::collections::HashMap<String, FileStamp>> {
    static V: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, FileStamp>>> =
        std::sync::OnceLock::new();
    V.get_or_init(Default::default)
}

fn forget(path: &str) {
    if let Ok(mut v) = verified().lock() {
        v.remove(path);
    }
}

#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
fn changed_since_verified(path: &str) -> AiKitError {
    AiKitError::Internal {
        reason: format!(
            "model file changed since it was verified against the catalog: {path}. A loaded model keeps using only the file it \
             verified, so it refuses this one, even if the new file has identical content. If the replacement was intentional, restart \
             the process (or create a new service) so the file is hashed against the catalog again; otherwise restore the original file"
        ),
    }
}

/// BLAKE3 of an open file from its start, streamed (model files can be gigabytes). The caller keeps the handle, so it can stamp the very
/// file that was hashed.
fn hash_handle(file: &mut File) -> std::io::Result<String> {
    let mut h = blake3::Hasher::new();
    h.update_reader(file)?;
    Ok(h.finalize().to_hex().to_string())
}

/// Hash an already-open handle and, when it equals `expected`, record **that handle's** stamp as the trusted one for `path`. Fails if the
/// handle changed while it was being hashed. The stamp comes from the handle that was hashed, never from a second look at the path, so a
/// path swapped for a bad file and back during the hash cannot make the bad file's stamp trusted.
fn hash_and_record_handle(path: &str, file: &mut File, expected: &str) -> Result<String> {
    let before = stamp_of(file)?;
    let actual = hash_handle(file)?;
    if stamp_of(file)? != before {
        forget(path);
        return Err(AiKitError::Internal {
            reason: format!("model file changed while it was being verified: {path}"),
        });
    }
    if actual == expected {
        if let Ok(mut v) = verified().lock() {
            v.insert(path.to_string(), before);
        }
    } else {
        // a previous good verification of this path no longer describes it
        forget(path);
    }
    Ok(actual)
}

#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
fn hash_and_record(path: &str, expected: &str) -> Result<String> {
    let mut file = File::open(path)?;
    hash_and_record_handle(path, &mut file, expected)
}

/// A model file opened once and checked, through the open handle, to be the file [`verify_model_files`] hashed: same size, modification
/// time, change time, device and inode. Read or map it through this handle and nothing else; a path that is swapped afterwards is not
/// consulted again. A file that was never verified in this process is refused.
///
/// What this gives you: a file replaced (rename) or rewritten (size, mtime, ctime change) after load is refused, without re-hashing
/// gigabytes per request. What it does not: see [`FileStamp`] for the undetectable same-tick in-place rewrite, and note that a mapping is
/// still undefined behaviour if the file is modified while mapped; the assumption, repeated at each mapping, is that nothing with write
/// access to the model directory touches a model file while it is in use.
#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
pub(crate) struct VerifiedFile {
    file: File,
    path: String,
    stamp: FileStamp,
}

#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
impl VerifiedFile {
    pub(crate) fn open(path: &str) -> Result<Self> {
        let recorded = verified().lock().ok().and_then(|v| v.get(path).copied());
        let Some(was) = recorded else {
            return Err(AiKitError::Internal {
                reason: format!(
                    "model file was never verified against the catalog in this process: {path} (load the model first)"
                ),
            });
        };
        let file = File::open(path)?;
        let now = stamp_of(&file)?;
        if now != was {
            return Err(changed_since_verified(path));
        }
        Ok(Self {
            file,
            path: path.to_string(),
            stamp: now,
        })
    }

    /// The handle, for `Read`/`Seek` consumers (GGUF parsing).
    #[cfg_attr(not(feature = "image"), allow(dead_code))]
    pub(crate) fn file_mut(&mut self) -> &mut File {
        &mut self.file
    }

    /// Fail if the handle changed since it was opened (a writer touched it while it was being read or mapped).
    pub(crate) fn recheck(&self) -> Result<()> {
        if stamp_of(&self.file)? == self.stamp {
            Ok(())
        } else {
            Err(changed_since_verified(&self.path))
        }
    }

    /// The whole file, read through the verified handle (for small files such as a tokenizer).
    pub(crate) fn read_all(mut self) -> Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(usize::try_from(self.stamp.len).unwrap_or(0));
        self.file.read_to_end(&mut buf)?;
        self.recheck()?;
        Ok(buf)
    }
}

/// Safetensors files mapped from verified handles. Each file is opened once, checked through that handle, and mapped from that same
/// handle, so a path swapped at any point cannot change what is mapped. Build a `VarBuilder` over them with [`Self::var_builder`]; the
/// mappings live as long as this value.
#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
pub(crate) struct VerifiedMaps {
    maps: Vec<memmap2::Mmap>,
}

#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
impl VerifiedMaps {
    pub(crate) fn open<P: AsRef<str>>(paths: &[P]) -> Result<Self> {
        let mut maps = Vec::with_capacity(paths.len());
        for p in paths {
            let f = VerifiedFile::open(p.as_ref())?;
            // SAFETY: mapping a file is unsafe because behaviour is undefined if the file is modified while it is mapped. Relied on:
            // (1) the handle mapped here is the one just checked against the stamp recorded when this process hashed the file against
            // the catalog (`VerifiedFile::open`), and it is checked again after the map call, so a swap by rename or a rewrite that
            // moved size, mtime or ctime is refused, and the path is never consulted again; (2) nothing with write access to the model
            // directory modifies the file while it is mapped. (2) is not enforced here: models are immutable OS resources and the
            // directory is not writable by other principals. A same-tick in-place rewrite by such a principal is not detectable (see
            // `FileStamp`).
            let map = unsafe { memmap2::Mmap::map(&f.file) }?;
            f.recheck()?;
            maps.push(map);
        }
        Ok(Self { maps })
    }

    /// A `VarBuilder` over every mapped file. A tensor name present in several files resolves to the last one, as Candle's own multi-file
    /// loader does.
    pub(crate) fn var_builder(
        &self,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<VarBuilder<'_>> {
        let files = self
            .maps
            .iter()
            .map(|m| SliceSafetensors::new(m))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(VarBuilder::from_backend(
            Box::new(Shards(files)),
            dtype,
            device.clone(),
        ))
    }
}

/// Several safetensors buffers behind one `VarBuilder` backend.
#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
struct Shards<'a>(Vec<SliceSafetensors<'a>>);

#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
impl Shards<'_> {
    fn load(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        match self.0.iter().rev().find(|s| s.get(name).is_ok()) {
            Some(s) => s.load(name, dev)?.to_dtype(dtype),
            None => Err(candle_core::Error::CannotFindTensor {
                path: name.to_string(),
            }
            .bt()),
        }
    }
}

#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
impl SimpleBackend for Shards<'_> {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let t = self.load(name, dtype, dev)?;
        if t.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: t.shape().clone(),
            }
            .bt())?
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.iter().any(|s| s.get(name).is_ok())
    }
}

/// Test fixtures point at weights that were not verified through a catalog entry: record their current stamp as the trusted one.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn trust_for_test(path: &str) {
    if let (Ok(f), Ok(mut v)) = (File::open(path), verified().lock()) {
        if let Ok(st) = stamp_of(&f) {
            v.insert(path.to_string(), st);
        }
    }
}

#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
/// Verify the main file and every extra file of a model against the catalog's hashes. Returns the main file's path.
pub(crate) fn verify_model_files(model: &ModelMetadata) -> Result<String> {
    let strip = |l: &str| l.strip_prefix("file://").unwrap_or(l).to_string();
    let main = strip(&model.location);
    let mut checks = vec![(main.clone(), model.hash.clone())];
    checks.extend(
        model
            .files
            .iter()
            .map(|f| (strip(&f.location), f.hash.clone())),
    );
    for (path, expected) in checks {
        let actual = hash_and_record(&path, &expected)?;
        if actual != expected {
            return Err(AiKitError::IntegrityCheckFailed {
                model_id: model.id.clone(),
                expected,
                actual,
            });
        }
    }
    Ok(main)
}

/// Open CUDA device `ordinal`, returning it with a label such as
/// `cuda:0 (NVIDIA GeForce RTX 3060)`. The error string says what to fix.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_device(ordinal: usize) -> std::result::Result<(Device, String), String> {
    let device = Device::new_cuda(ordinal)
        .map_err(|e| format!("CUDA device {ordinal} is not usable: {e}"))?;
    let name = device
        .as_cuda_device()
        .map_err(|e| e.to_string())?
        .cuda_stream()
        .context()
        .name()
        .unwrap_or_else(|_| "unknown GPU".to_string());
    Ok((device, format!("cuda:{ordinal} ({name})")))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn cuda_device(ordinal: usize) -> std::result::Result<(Device, String), String> {
    Err(format!(
        "CUDA device {ordinal} requested, but this ai-kit was built without the `cuda` \
         feature (rebuild with `--features cuda`)"
    ))
}

/// MB of weights + headroom the model needs on a GPU. The catalog's declared
/// `vram_required_mb` is honoured as a floor.
// Only the CUDA path calls this, but it is plain arithmetic, so it is compiled
// (and unit-tested) in every build rather than hidden behind the feature.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn vram_needed_mb(model: &ModelMetadata) -> u64 {
    // A model that streams its weights (Z-Image) never holds them all at once, so the sum of its file sizes overstates what it
    // needs. Its catalog entry declares the measured peak in `custom.vram_peak_mb`, which replaces the file-size estimate.
    if let Some(peak) = model
        .metadata
        .custom
        .get("vram_peak_mb")
        .and_then(|v| v.parse::<u64>().ok())
    {
        return peak.max(u64::from(model.metadata.vram_required_mb));
    }
    let weights_mb = model.total_size_bytes().div_ceil(1 << 20);
    let overhead =
        ((weights_mb as f64 * VRAM_OVERHEAD_FRACTION) as u64).max(VRAM_OVERHEAD_FLOOR_MB);
    (weights_mb + overhead).max(u64::from(model.metadata.vram_required_mb))
}

/// Fail hard (`InferenceOom`, no spill to the CPU, no smaller quantisation) if a
/// model cannot fit in the free memory of the chosen GPU. No-op on the CPU.
#[cfg(feature = "cuda")]
pub(crate) fn check_vram(device: &Device, model: &ModelMetadata) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let ctx = device
        .as_cuda_device()
        .map_err(|e| AiKitError::Internal {
            reason: e.to_string(),
        })?
        .cuda_stream()
        .context()
        .clone();
    let (free, _total) = ctx.mem_get_info().map_err(|e| AiKitError::Internal {
        reason: format!("cannot query free GPU memory: {e}"),
    })?;
    let free_mb = (free >> 20) as u64;
    let need_mb = vram_needed_mb(model);
    if need_mb > free_mb {
        return Err(AiKitError::InferenceOom {
            model_id: model.id.clone(),
            required_mb: u32::try_from(need_mb).unwrap_or(u32::MAX),
            available_mb: u32::try_from(free_mb).unwrap_or(u32::MAX),
        });
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn check_vram(_device: &Device, _model: &ModelMetadata) -> Result<()> {
    Ok(())
}

/// One line per usable CUDA device (`cuda:0 NVIDIA GeForce RTX 3060, 11.4/12.0 GB free`),
/// or a single line saying why there are none. For `ai-kit-cli devices`.
pub fn describe_cuda_devices() -> Vec<String> {
    #[cfg(feature = "cuda")]
    {
        let mut out = Vec::new();
        for ordinal in 0..16 {
            let Ok(device) = Device::new_cuda(ordinal) else {
                break;
            };
            let Ok(cuda) = device.as_cuda_device() else {
                break;
            };
            let ctx = cuda.cuda_stream().context().clone();
            let name = ctx.name().unwrap_or_else(|_| "unknown GPU".to_string());
            let mem = ctx
                .mem_get_info()
                .map(|(free, total)| {
                    format!("{:.1}/{:.1} GB free", free as f64 / 1e9, total as f64 / 1e9)
                })
                .unwrap_or_else(|e| format!("memory unknown ({e})"));
            out.push(format!("cuda:{ordinal} {name}, {mem}"));
        }
        if out.is_empty() {
            out.push("no usable CUDA device found (driver missing, or no GPU)".to_string());
        }
        out
    }
    #[cfg(not(feature = "cuda"))]
    {
        vec!["built without the `cuda` feature; only `cpu` is available".to_string()]
    }
}

#[cfg_attr(not(feature = "image"), allow(dead_code))]
/// True if an error message is a GPU out-of-memory (Candle surfaces it as text, e.g. `CUDA_ERROR_OUT_OF_MEMORY`).
pub(crate) fn looks_like_oom(text: &str) -> bool {
    let t = text.to_ascii_uppercase();
    t.contains("OUT_OF_MEMORY") || t.contains("OUT OF MEMORY")
}

/// Free GPU memory in MB, if `device` is a CUDA device that can be queried.
#[cfg(feature = "cuda")]
#[cfg_attr(not(feature = "image"), allow(dead_code))]
fn free_vram_mb(device: &Device) -> Option<u64> {
    let ctx = device
        .as_cuda_device()
        .ok()?
        .cuda_stream()
        .context()
        .clone();
    ctx.mem_get_info().ok().map(|(free, _)| (free >> 20) as u64)
}
#[cfg(not(feature = "cuda"))]
#[cfg_attr(not(feature = "image"), allow(dead_code))]
fn free_vram_mb(_device: &Device) -> Option<u64> {
    None
}

#[cfg_attr(not(feature = "image"), allow(dead_code))]
/// Turn an `Internal` error that is really a GPU out-of-memory into a typed `InferenceOom`, so callers and the wire contract can tell
/// "does not fit" from "broken". `required_mb` is 0 because the real need is not known at that point; `available_mb` is the free
/// memory right now when it can be read. Any other error is returned unchanged.
pub(crate) fn promote_oom(e: AiKitError, model_id: &str, device: Option<&Device>) -> AiKitError {
    match e {
        AiKitError::Internal { ref reason } if looks_like_oom(reason) => AiKitError::InferenceOom {
            model_id: model_id.to_string(),
            required_mb: 0,
            available_mb: device
                .and_then(free_vram_mb)
                .map_or(0, |m| u32::try_from(m).unwrap_or(u32::MAX)),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelMetadataExtended;
    use std::collections::HashMap;

    fn model(size_bytes: u64, vram_required_mb: u32) -> ModelMetadata {
        ModelMetadata {
            id: "m".to_string(),
            version: "1.0.0".to_string(),
            os_arch: vec!["x86_64".to_string()],
            quantization_variants: vec!["q4_k_m".to_string()],
            default_quantization: "q4_k_m".to_string(),
            hash: "0".to_string(),
            size_bytes,
            location: "file:///x".to_string(),
            metadata: ModelMetadataExtended {
                parameters: 1,
                context_window: 2048,
                vram_required_mb,
                inference_latency_ms: 0,
                framework: String::new(),
                custom: HashMap::new(),
            },
            files: Vec::new(),
        }
    }

    /// Write `bytes` to `name` in `dir`, let the filesystem's timestamp tick pass, and verify it against the hash of `bytes`.
    /// Returns the path. The pause matters: without it a rewrite made straight afterwards can land in the same tick as the stamp.
    fn verified_file(dir: &std::path::Path, name: &str, bytes: &[u8]) -> String {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let p = path.to_str().unwrap().to_string();
        let mut m = model(1, 1);
        m.location = format!("file://{p}");
        m.hash = blake3::hash(bytes).to_hex().to_string();
        verify_model_files(&m).unwrap();
        p
    }

    fn refused(r: Result<VerifiedFile>, what: &str) {
        match r {
            Err(AiKitError::Internal { reason }) => {
                assert!(
                    reason.contains("changed since it was verified"),
                    "{what}: {reason}"
                );
                assert!(
                    reason.contains("restart"),
                    "{what}: the error says what to do: {reason}"
                );
            }
            Err(e) => panic!("{what}: wrong error {e}"),
            Ok(_) => panic!("{what}: was accepted"),
        }
    }

    #[test]
    fn a_never_verified_file_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("w.safetensors");
        std::fs::write(&path, b"the weights").unwrap();
        let r = VerifiedFile::open(path.to_str().unwrap());
        assert!(
            matches!(r, Err(AiKitError::Internal { reason }) if reason.contains("never verified"))
        );
    }

    #[test]
    fn a_verified_file_that_is_untouched_opens_and_reads_back_what_was_hashed() {
        let d = tempfile::tempdir().unwrap();
        let p = verified_file(d.path(), "w", b"the weights");
        assert_eq!(
            VerifiedFile::open(&p).unwrap().read_all().unwrap(),
            b"the weights"
        );
    }

    #[test]
    fn a_file_swapped_for_a_same_size_file_by_rename_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let p = verified_file(d.path(), "w", b"the weights");
        let other = d.path().join("other");
        std::fs::write(&other, b"evil weight").unwrap(); // same length
        std::fs::rename(&other, &p).unwrap();
        refused(VerifiedFile::open(&p), "rename swap");
    }

    #[test]
    fn a_same_size_in_place_rewrite_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let p = verified_file(d.path(), "w", b"the weights");
        std::fs::write(&p, b"evil weight").unwrap(); // same inode, same length
        refused(VerifiedFile::open(&p), "same-size rewrite");
    }

    #[cfg(unix)]
    #[test]
    fn restoring_the_modification_time_after_a_rewrite_does_not_hide_it() {
        let d = tempfile::tempdir().unwrap();
        let p = verified_file(d.path(), "w", b"the weights");
        let before = std::fs::metadata(&p).unwrap().modified().unwrap();
        let f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
        std::io::Write::write_all(&mut &f, b"evil weight").unwrap();
        f.set_modified(before).unwrap(); // size and mtime are now exactly as they were; only ctime moved
        assert_eq!(std::fs::metadata(&p).unwrap().modified().unwrap(), before);
        refused(VerifiedFile::open(&p), "mtime restored");
    }

    #[test]
    fn a_file_rewritten_with_a_different_size_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let p = verified_file(d.path(), "w", b"the weights");
        std::fs::write(&p, b"longer replacement bytes").unwrap();
        refused(VerifiedFile::open(&p), "resized");
    }

    #[test]
    fn re_verifying_changed_content_trusts_the_new_stamp_and_a_wrong_hash_forgets_the_old_one() {
        let d = tempfile::tempdir().unwrap();
        let p = verified_file(d.path(), "w", b"the weights");
        std::fs::write(&p, b"evil weight").unwrap();
        refused(VerifiedFile::open(&p), "changed");
        let mut m = model(1, 1);
        m.location = format!("file://{p}");
        m.hash = blake3::hash(b"evil weight").to_hex().to_string();
        verify_model_files(&m).unwrap();
        VerifiedFile::open(&p).expect("re-verified");
        m.hash = "0".into();
        assert!(verify_model_files(&m).is_err());
        assert!(
            VerifiedFile::open(&p).is_err(),
            "a failed verification forgets the earlier good one"
        );
    }

    #[test]
    fn the_recorded_stamp_is_the_hashed_handles_even_if_the_path_is_swapped_during_the_hash() {
        // The path held the good file when it was opened for hashing, then a bad file of the same size was put at the path (and the
        // hash read, in the old by-path code, would have seen either). The stamp that gets trusted must be the hashed handle's.
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("w");
        std::fs::write(&path, b"the weights").unwrap();
        let p = path.to_str().unwrap();
        let mut good = File::open(&path).unwrap();
        let bad = d.path().join("bad");
        std::fs::write(&bad, b"evil weight").unwrap();
        std::fs::rename(&bad, &path).unwrap();
        let expected = blake3::hash(b"the weights").to_hex().to_string();
        assert_eq!(
            hash_and_record_handle(p, &mut good, &expected).unwrap(),
            expected
        );
        refused(
            VerifiedFile::open(p),
            "the swapped-in file must not inherit the good file's trust",
        );
    }

    #[test]
    fn maps_open_only_verified_unchanged_files_and_read_the_handle_they_checked() {
        let d = tempfile::tempdir().unwrap();
        let t = Tensor::new(&[1f32, 2.0, 3.0], &Device::Cpu).unwrap();
        let file = d.path().join("a.safetensors");
        candle_core::safetensors::save(&HashMap::from([("t".to_string(), t)]), &file).unwrap();
        let bytes = std::fs::read(&file).unwrap();
        let p = verified_file(d.path(), "a.safetensors", &bytes);
        let maps = VerifiedMaps::open(&[p.as_str()]).unwrap();
        let vb = maps.var_builder(DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            vb.get(3, "t").unwrap().to_vec1::<f32>().unwrap(),
            [1.0, 2.0, 3.0]
        );
        assert!(vb.get(4, "t").is_err(), "shape is checked");
        assert!(vb.get(3, "missing").is_err());
        drop(vb);
        // swapped by rename for a same-size file: refused, and the first mapping is unaffected
        let other = d.path().join("other");
        std::fs::write(&other, &bytes).unwrap();
        std::fs::rename(&other, &p).unwrap();
        assert!(matches!(
            VerifiedMaps::open(&[p.as_str()]),
            Err(AiKitError::Internal { reason }) if reason.contains("changed since it was verified")
        ));
        assert!(VerifiedMaps::open(&["/nonexistent/x"]).is_err());
    }

    #[test]
    fn a_wrong_hash_is_still_an_integrity_failure_and_records_nothing_it_trusts() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("w2.safetensors");
        std::fs::write(&path, b"abc").unwrap();
        let mut m = model(1, 1);
        m.location = format!("file://{}", path.to_str().unwrap());
        m.hash = "0".into();
        assert!(matches!(
            verify_model_files(&m),
            Err(AiKitError::IntegrityCheckFailed { .. })
        ));
        assert!(
            VerifiedFile::open(path.to_str().unwrap()).is_err(),
            "a file whose hash did not match is never trusted"
        );
    }

    #[test]
    fn vram_estimate_adds_proportional_headroom() {
        // 4000 MB of weights: 15% = 600 MB of headroom, above the 384 MB floor.
        let m = model(4000 << 20, 1);
        assert_eq!(vram_needed_mb(&m), 4600);
    }

    #[test]
    fn vram_estimate_has_a_floor_for_small_models() {
        // 100 MB of weights: 15% = 15 MB, so the 384 MB floor applies.
        assert_eq!(vram_needed_mb(&model(100 << 20, 1)), 100 + 384);
    }

    #[test]
    fn a_gpu_out_of_memory_message_becomes_a_typed_error_and_nothing_else_does() {
        let oom = AiKitError::Internal {
            reason: "z_image vae decode: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
                .into(),
        };
        assert!(matches!(
            promote_oom(oom, "m", None),
            AiKitError::InferenceOom { .. }
        ));
        let other = AiKitError::Internal {
            reason: "shape mismatch".into(),
        };
        assert!(matches!(
            promote_oom(other, "m", None),
            AiKitError::Internal { .. }
        ));
        let typed = AiKitError::ModelNotFound {
            model_id: "m".into(),
        };
        assert!(matches!(
            promote_oom(typed, "m", None),
            AiKitError::ModelNotFound { .. }
        ));
    }

    #[test]
    fn a_streaming_model_declares_its_measured_peak_instead_of_its_file_sizes() {
        let mut m = model(12_000 << 20, 1);
        m.metadata
            .custom
            .insert("vram_peak_mb".to_string(), "8500".to_string());
        assert_eq!(vram_needed_mb(&m), 8500);
        m.metadata.vram_required_mb = 9000; // the declared floor still wins if it is higher
        assert_eq!(vram_needed_mb(&m), 9000);
    }

    #[test]
    fn vram_estimate_honours_the_catalog_declared_floor() {
        // The catalog says it needs 9000 MB, more than the file-size estimate.
        assert_eq!(vram_needed_mb(&model(1000 << 20, 9000)), 9000);
    }
}
