# Evaluation helpers (sane defaults, two targets):
#   make bench-lang      # semantic language eval on lang-eval-dev @ k=10
#   make bench-cluster   # clustering eval on cluster-stress-dev
#
# Optional: override config file via CONFIG, e.g.:
#   make bench-lang CONFIG=.chunkhound.json

.PHONY: bench-lang bench-cluster dev dev-release lint typecheck test rust-check rust-test

# DuckDB is linked dynamically via DUCKDB_DOWNLOAD_LIB + a self-relative RPATH
# baked in at link time ($ORIGIN on Linux, @loader_path on macOS) -- see
# RUST_COMMANDS in AGENTS.md. scripts/copy_duckdb_runtime.py then places the
# downloaded library next to the compiled extension so that RPATH has
# something to find.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	RUST_RPATH_FLAG := -Wl,-rpath,@loader_path
else
	RUST_RPATH_FLAG := -Wl,-rpath,$$ORIGIN
endif
RUST_DUCKDB_ENV := DUCKDB_DOWNLOAD_LIB=1 RUSTFLAGS='-C link-arg=$(RUST_RPATH_FLAG)'

bench-lang:
	uv run python -m chunkhound.tools.eval_search \
		--bench-id lang-eval-dev \
		--mode mixed \
		--search-mode semantic \
		--languages all \
		--k 10 \
		$(if $(CONFIG),--config $(CONFIG),) \
		--output .chunkhound/benches/lang-eval-dev/eval_semantic_k10.json

bench-cluster:
	uv run python -m chunkhound.tools.eval_cluster \
		--bench-id cluster-stress-dev \
		$(if $(CONFIG),--config $(CONFIG),) \
		--output .chunkhound/benches/cluster-stress-dev/cluster_eval.json

dev:
	$(RUST_DUCKDB_ENV) uv run maturin develop
	uv run python scripts/copy_duckdb_runtime.py
	uv run pytest tests/test_smoke.py -v -n auto

dev-release:
	rm -rf target/wheels/
	# maturin's wheel-repair step renames the linked libduckdb.so to a hashed
	# name IN PLACE inside cargo's own cached release artifact; a rerun that
	# reuses that cache (no source changes) then fails looking for a hashed
	# name from the previous wheel. Force a clean relink every time.
	cargo clean -p chunkhound_native --release
	$(RUST_DUCKDB_ENV) uv run maturin build --release --out target/wheels/
	uv run python scripts/install_native.py
	uv run python scripts/copy_duckdb_runtime.py
	uv run pytest tests/test_smoke.py -v -n auto

lint:
	uv run ruff check chunkhound

typecheck:
	uv run mypy chunkhound

test:
	uv run pytest tests/ -v

rust-check:
	cargo fmt --check
	DUCKDB_DOWNLOAD_LIB=1 cargo clippy --all-targets -- -D warnings

rust-test:
	DUCKDB_DOWNLOAD_LIB=1 cargo test
