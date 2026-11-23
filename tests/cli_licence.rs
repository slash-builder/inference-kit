//! The licence rule is the same for every command that loads a model the user names on the command line.
//!
//! These run the real `ai-kit-cli` binary against a catalog whose weights do not exist: a command that gets past the licence check
//! fails later with a different message, which is how "refused for the licence" is told from "got as far as loading".
#![cfg(feature = "candle")]

use std::process::{Command, Output};

const ENTRY: &str = r#"  - id: "{id}"
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
        kind: "{kind}"
        license: "{license}"
"#;

fn catalog() -> (tempfile::TempDir, String) {
    let mut yaml = String::from("version: \"1.0\"\nmodels:\n");
    for (kind, short) in [
        ("text_generation", "text"),
        ("embedding", "embed"),
        ("image_generation", "image"),
    ] {
        for (prefix, license) in [
            ("nc", "CC-BY-NC-4.0"),
            ("unknown", "GPL-3.0"),
            ("ok", "Apache-2.0"),
        ] {
            yaml.push_str(
                &ENTRY
                    .replace("{id}", &format!("{prefix}-{short}"))
                    .replace("{kind}", kind)
                    .replace("{license}", license),
            );
        }
    }
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("catalog.yaml");
    std::fs::write(&path, yaml).unwrap();
    let p = path.to_str().unwrap().to_string();
    (d, p)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ai-kit-cli"))
        .args(args)
        .output()
        .unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn refused(o: &Output) -> bool {
    o.status.code() == Some(2) && stderr(o).contains("not known to permit commercial use")
}

/// Each way of naming a model, for a model id of the matching kind.
fn invocations<'a>(
    catalog: &'a str,
    id_text: &'a str,
    id_embed: &'a str,
    id_image: &'a str,
) -> Vec<(&'static str, Vec<&'a str>)> {
    // never written: every case here stops before generating anything
    let out = "/nonexistent-dir/ai-kit-licence-test.png";
    vec![
        (
            "infer --model",
            vec![
                "infer",
                "--catalog",
                catalog,
                "--model",
                id_text,
                "--prompt",
                "hi",
            ],
        ),
        (
            "bench --model",
            vec![
                "bench",
                "--catalog",
                catalog,
                "--model",
                id_text,
                "--prompt",
                "hi",
                "--repeat",
                "1",
            ],
        ),
        (
            "embed --model",
            vec![
                "embed",
                "--catalog",
                catalog,
                "--model",
                id_embed,
                "--text",
                "hi",
            ],
        ),
        (
            "imagine --model",
            vec![
                "imagine",
                "--catalog",
                catalog,
                "--model",
                id_image,
                "--prompt",
                "hi",
                "--out",
                out,
            ],
        ),
        (
            "route --local-model",
            vec![
                "route",
                "--catalog",
                catalog,
                "--prompt",
                "hi",
                "--local-model",
                id_text,
            ],
        ),
        (
            "route --local-model --explain",
            vec![
                "route",
                "--catalog",
                catalog,
                "--prompt",
                "hi",
                "--explain",
                "--local-model",
                id_text,
            ],
        ),
        (
            "route --image-model",
            vec![
                "route",
                "--catalog",
                catalog,
                "--prompt",
                "hi",
                "--image-model",
                id_image,
            ],
        ),
        (
            "route --embed-model",
            vec![
                "route",
                "--catalog",
                catalog,
                "--prompt",
                "hi",
                "--embed-model",
                id_embed,
            ],
        ),
    ]
}

#[test]
fn every_named_model_path_refuses_a_non_commercial_model() {
    let (_d, cat) = catalog();
    for (name, args) in invocations(&cat, "nc-text", "nc-embed", "nc-image") {
        let o = run(&args);
        assert!(
            refused(&o),
            "{name} was not refused: {:?} {}",
            o.status,
            stderr(&o)
        );
    }
}

#[test]
fn every_named_model_path_refuses_a_licence_it_does_not_recognise() {
    let (_d, cat) = catalog();
    for (name, args) in invocations(&cat, "unknown-text", "unknown-embed", "unknown-image") {
        let o = run(&args);
        assert!(
            refused(&o),
            "{name} was not refused: {:?} {}",
            o.status,
            stderr(&o)
        );
    }
}

#[test]
fn allow_noncommercial_lets_every_named_model_path_past_the_licence_check() {
    let (_d, cat) = catalog();
    for (name, mut args) in invocations(&cat, "nc-text", "nc-embed", "nc-image") {
        args.push("--allow-noncommercial");
        let o = run(&args);
        assert!(
            !refused(&o),
            "{name} was refused despite --allow-noncommercial: {}",
            stderr(&o)
        );
    }
}

#[test]
fn a_permissive_model_is_not_refused_on_any_path() {
    let (_d, cat) = catalog();
    for (name, args) in invocations(&cat, "ok-text", "ok-embed", "ok-image") {
        let o = run(&args);
        assert!(!refused(&o), "{name} was refused: {}", stderr(&o));
    }
}
