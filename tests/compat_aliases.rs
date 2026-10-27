//! The 0.1.x names still work (deprecated) so downstream code can migrate at its own pace.
#![allow(deprecated)]

#[test]
fn the_old_error_name_is_an_alias_of_error() {
    let old: inference_kit::AiKitError = inference_kit::Error::ModelNotFound {
        model_id: "m".into(),
    };
    // Same type: it can be passed where `Error` is expected, and matched through either name.
    let new: inference_kit::Error = old;
    assert!(matches!(
        new,
        inference_kit::AiKitError::ModelNotFound { .. }
    ));
}

#[test]
fn the_legacy_environment_prefix_is_still_read() {
    std::env::set_var("AI_KIT_COMPAT_TEST", "1");
    assert_eq!(inference_kit::env::var("COMPAT_TEST").as_deref(), Some("1"));
    std::env::set_var("INFERENCE_KIT_COMPAT_TEST", "2");
    assert_eq!(inference_kit::env::var("COMPAT_TEST").as_deref(), Some("2"));
}
