//! An API-key header value marked sensitive, so `{:?}` of a request or its header map prints `Sensitive` instead of the key.
//! (reqwest's `bearer_auth` already does this for `Authorization`; the vendors that use a custom key header need it done by hand.)
//! Shared by every adapter that sends a key in a custom header.

use reqwest::header::HeaderValue;

/// `None` if `secret` is not a valid header value (the caller maps that to its own auth error; the key is never put in it).
pub(crate) fn sensitive_header_value(secret: &str) -> Option<HeaderValue> {
    let mut v = HeaderValue::from_str(secret).ok()?;
    v.set_sensitive(true);
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::sensitive_header_value;

    #[test]
    fn an_api_key_header_is_sensitive_and_does_not_show_in_debug_output() {
        const KEY: &str = "sk-TEST-ONLY-header-key";
        let req = reqwest::Client::new()
            .get("http://127.0.0.1:9/")
            .header("x-api-key", sensitive_header_value(KEY).unwrap())
            .build()
            .unwrap();
        let shown = format!("{:?} {:?}", req.headers(), req);
        assert!(!shown.contains(KEY), "{shown}");
        assert!(shown.contains("Sensitive"), "{shown}");
        // it is still the real value on the wire
        let v = req.headers().get("x-api-key").unwrap();
        assert!(v.is_sensitive());
        assert_eq!(v.to_str().unwrap(), KEY);
    }

    #[test]
    fn a_key_that_is_not_a_valid_header_value_is_refused() {
        assert!(sensitive_header_value("bad\nkey").is_none());
    }
}
