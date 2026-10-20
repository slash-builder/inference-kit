//! The one place YAML is parsed.
//!
//! `serde_saphyr` with the strict preset the studio standard requires (`rust-and-crates-io-standard.md`, section 11):
//! the YAML 1.1 words `y`, `yes`, `no`, `on` and `off` stay strings rather than turning into booleans; a duplicate key,
//! a merge key (`<<`) or an unknown tag is an error; `.inf` and `.nan` are refused (a limit that becomes null or infinity
//! fails open); and an error never echoes a snippet of the document.

use serde::de::DeserializeOwned;
use serde_saphyr::{DuplicateKeyPolicy, MergeKeyPolicy, NonFiniteFloatPolicy, Options};

/// Parse `s` as a single YAML document with strict options.
pub(crate) fn from_str<T: DeserializeOwned>(s: &str) -> Result<T, serde_saphyr::Error> {
    serde_saphyr::from_str_with_options(s, options())
}

fn options() -> Options {
    let mut o = Options::default();
    o.strict_booleans = true;
    o.duplicate_keys = DuplicateKeyPolicy::Error;
    o.merge_keys = MergeKeyPolicy::Error;
    o.reject_unsupported_tags = true;
    o.non_finite_float_policy = NonFiniteFloatPolicy::Reject;
    o.with_snippet = false;
    o
}

#[cfg(test)]
mod tests {
    use super::from_str;
    use std::collections::HashMap;

    #[test]
    fn yes_and_no_are_not_booleans() {
        let m: HashMap<String, String> = from_str("country: no\nflag: yes\n").unwrap();
        assert_eq!(m["country"], "no");
        assert_eq!(m["flag"], "yes");
    }

    #[test]
    fn merge_keys_and_non_finite_floats_are_errors() {
        let m: Result<HashMap<String, HashMap<String, u32>>, _> =
            from_str("base: &b {x: 1}\nd:\n  <<: *b\n");
        assert!(m.is_err());
        // The policy covers values read into a dynamic tree; a field typed `f64` is the caller's to bound.
        let f: Result<serde_json::Value, _> = from_str("limit: .inf\n");
        assert!(f.is_err());
    }

    #[test]
    fn duplicate_keys_are_an_error() {
        let r: Result<HashMap<String, u32>, _> = from_str("a: 1\na: 2\n");
        assert!(r.is_err());
    }
}
