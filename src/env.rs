//! Environment variables read by Inference Kit.
//!
//! Every variable is named `INFERENCE_KIT_<NAME>`. The 0.1.x names, `AI_KIT_<NAME>`, are
//! still read as a **deprecated fallback** (a warning is logged the first time each one is
//! used) and will stop being read in a later release.
//!
//! ```
//! // Reads INFERENCE_KIT_DEVICE, then the deprecated AI_KIT_DEVICE.
//! let device = inference_kit::env::var("DEVICE");
//! # let _ = device;
//! ```

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// The prefix of every current variable.
pub const PREFIX: &str = "INFERENCE_KIT_";
/// The deprecated 0.1.x prefix, read only as a fallback.
pub const LEGACY_PREFIX: &str = "AI_KIT_";

/// Read `INFERENCE_KIT_<name>`, falling back to the deprecated `AI_KIT_<name>`.
///
/// `name` is the part after the prefix, for example `"DEVICE"`. Returns `None` when neither
/// is set or the value is not valid Unicode.
pub fn var(name: &str) -> Option<String> {
    if let Ok(v) = std::env::var(format!("{PREFIX}{name}")) {
        return Some(v);
    }
    let legacy = format!("{LEGACY_PREFIX}{name}");
    let v = std::env::var(&legacy).ok()?;
    warn_once(&legacy, name);
    Some(v)
}

fn warn_once(legacy: &str, name: &str) {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let first = seen
        .lock()
        .map(|mut s| s.insert(legacy.to_string()))
        .unwrap_or(true);
    if first {
        tracing::warn!(
            "{legacy} is deprecated and will stop being read in a later release; set {PREFIX}{name} instead"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each test uses its own variable name so parallel tests never share state.

    #[test]
    fn the_new_name_wins_over_the_legacy_name() {
        std::env::set_var("INFERENCE_KIT_T_BOTH", "new");
        std::env::set_var("AI_KIT_T_BOTH", "old");
        assert_eq!(var("T_BOTH").as_deref(), Some("new"));
    }

    #[test]
    fn the_legacy_name_is_still_read() {
        std::env::set_var("AI_KIT_T_LEGACY", "old");
        assert_eq!(var("T_LEGACY").as_deref(), Some("old"));
    }

    #[test]
    fn unset_is_none() {
        assert_eq!(var("T_NEITHER"), None);
    }
}
