.PHONY: build test fmt lint clean help all fetch-model package-check package-check-selftest

help:
	@echo "AI Kit — Makefile targets"
	@echo ""
	@echo "  make build        Build release binary"
	@echo "  make test         Run all tests"
	@echo "  make fmt          Format code"
	@echo "  make lint         Run clippy linter"
	@echo "  make check        Run fmt + lint + test (pre-commit)"
	@echo "  make clean        Remove build artifacts"
	@echo "  make all          Build + test + fmt + lint"
	@echo "  make package-check  Fail if the published crate would contain cargo config,"
	@echo "                    credentials, env files or build output (names only)"
	@echo "  make fetch-model  Download a small model for local --features candle"
	@echo "                    testing (POC only). MODEL=name to pick, default tinyllama."
	@echo ""

all: build test fmt lint

build:
	cargo build --release

test:
	cargo test

fetch-model:
	./scripts/fetch-model.sh $(MODEL)

fmt:
	cargo fmt

lint:
	cargo clippy --all-targets -- -D warnings

# The gate. Run by CI on every build and again, hard, before every publish.
package-check:
	./scripts/check-package.sh

# Proves the gate fails closed: planted forbidden names must be rejected, and
# source files that merely contain the word "credentials" must not be.
package-check-selftest:
	@if printf 'Cargo.toml\nsrc/lib.rs\n.cargo/credentials.toml\n' | ./scripts/check-package.sh --stdin >/dev/null 2>&1; then \
	  echo "selftest FAILED: .cargo/credentials.toml was not rejected"; exit 1; fi
	@if printf 'Cargo.toml\nsrc/lib.rs\ncredentials.toml\n' | ./scripts/check-package.sh --stdin >/dev/null 2>&1; then \
	  echo "selftest FAILED: credentials.toml was not rejected"; exit 1; fi
	@printf 'Cargo.toml\nsrc/credentials.rs\nsrc/credentials/conformance.rs\n' | ./scripts/check-package.sh --stdin >/dev/null
	@echo "package-check selftest OK"

check: fmt lint test
	@echo "✓ All checks passed"

clean:
	cargo clean
