//! The one place YAML is parsed.
//!
//! `serde_saphyr` with the strict preset the studio standard requires
//! (`rust-and-crates-io-standard.md`): the YAML 1.1 words `y`, `yes`, `no`, `on` and `off` stay strings
//! rather than turning into booleans, and duplicate keys are an error instead of last-one-wins.

use serde::de::DeserializeOwned;

/// Parse `s` as a single YAML document with strict options.
pub(crate) fn from_str<T: DeserializeOwned>(s: &str) -> Result<T, serde_saphyr::Error> {
    let mut options = serde_saphyr::Options::default();
    options.strict_booleans = true;
    serde_saphyr::from_str_with_options(s, options)
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
    fn duplicate_keys_are_an_error() {
        let r: Result<HashMap<String, u32>, _> = from_str("a: 1\na: 2\n");
        assert!(r.is_err());
    }
}
