# Contributing to Inference Kit

Thank you for your interest in contributing to Inference Kit! This document provides guidelines for contributions.

## Code of Conduct

Please read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## Before You Start

- Read the [README](README.md) and `CHANGELOG.md` to see what the crate does and what is still experimental.
- Cloud features are opt-in and send data to third parties; a change that widens what can leave the machine needs a clear justification in the pull request.
- The default build must stay local-only (`default = []`, no network code compiled in).

## Development Workflow

### Setup

```bash
git clone https://github.com/slash-builder/inference-kit.git
cd inference-kit
cargo build
cargo test
```

### Code Style

```bash
cargo fmt
cargo clippy -- -D warnings
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature").
- Use imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line to 72 characters or less.
- Reference issues and decisions (e.g., "Refs: #12", "Fixes: #42").

### Testing

All changes must include tests:

```bash
cargo test
```

### Documentation

- Document public APIs with doc comments.
- Update README.md if adding user-facing features.
- Note any open questions or TODOs related to pending decisions.

## Open Design Questions

Work that depends on a decision that has not been made yet should:

1. Add a `TODO` comment that states the open question in plain words (and links the issue, if there is one).
2. Stub out the feature with a placeholder or an explicit error.
3. Say so in the pull request description.

### Example

```rust
// TODO: pre-load strategy is undecided (see the tracking issue).
// Once it is decided, a `preload_all()` method may be added.
async fn load_model(&self, model_id: &str) -> Result<()> {
    // ... lazy-load implementation
}
```

## Invariants (Never Break)

The 8 locked invariants are inviolable:

1. Models are OS resources (shipped signed, immutable).
2. No weights over network (MVP).
3. Trait-based API (no framework-specific exports).
4. KV cache isolation (per-session, never cross-app).
5. Audit trail immutable (always-on).
6. Score Kit gates capability (single source of truth).
7. Versions lock at install (no auto-upgrade).
8. Inference-only (no training, fine-tuning).

Any PR that conflicts with these will be rejected.

## Integration Points

When touching integration surfaces, coordinate with:

- **Message Kit** — async envelope bindings
- **Storage Kit** — model artifact persistence
- **Score Kit** — capability grants
- **Service Kit** — observability
- **Spec Kit** — conformance validation
- **Device Kit** — vRAM/quantization hints

## Filing Issues

Include:

- Clear title referencing the component (e.g., "registry: handle missing catalog.yaml").
- Any open design question the issue depends on (if any).
- Reproduction steps (if a bug).
- Links to related issues or specs.

## Pull Request Process

1. Fork and create a feature branch.
2. Commit with clear messages.
3. Ensure all tests pass (`cargo test` + `cargo clippy`).
4. Update documentation and AGENT.md if needed.
5. Open a PR with a clear description.
6. Wait for review from maintainers.

## Questions?

Reach out to the Inference Kit PM (TBD once project is elevated). Until then, file an issue or PR.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
