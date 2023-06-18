watch:
	cargo watch -x check -x test

tests:
	cargo test --all-targets --all-features

lint:
	cargo clippy --all-targets --all-features -- -D warnings

format:
	cargo fmt

audit:
	cargo audit

coverage:
	cargo tarpaulin --ignore-tests

pipeline:
	cargo check && \
	cargo test --all-targets --all-features && \
	cargo clippy --all-targets --all-features -- -D warnings && \
	cargo fmt && \
	cargo audit && \
	cargo tarpaulin --ignore-tests
