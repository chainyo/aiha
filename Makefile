lint:
	cargo clippy --all-targets --all-features -- -D warnings

tests:
	cargo test --all-targets --all-features
