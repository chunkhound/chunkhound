# Evaluation helpers (sane defaults, two targets):
#   make bench-lang      # semantic language eval on lang-eval-dev @ k=10
#   make bench-cluster   # clustering eval on cluster-stress-dev
#
# Optional: override config file via CONFIG, e.g.:
#   make bench-lang CONFIG=.chunkhound.json

.PHONY: bench-lang bench-cluster dev dev-release lint typecheck test rust-check rust-test

# DuckDB is linked dynamically via DUCKDB_DOWNLOAD_LIB + a self-relative RPATH
# baked in at link time ($ORIGIN on Linux, @loader_path on macOS).
# scripts/copy_duckdb_runtime.py then places the downloaded library next to
# the compiled extension so that RPATH has something to find.
#
# Air-gapped builds (no GitHub access for the DuckDB download): link a static
# libduckdb you already have instead -- DUCKDB_LIB_DIR=<dir> DUCKDB_STATIC=1
# with RUSTFLAGS='-C link-arg=-lstdc++' (the static .a's C++ exception code
# lives in libstdc++.so; without it the .so builds but fails at import with
# "undefined symbol: __gxx_personality_v0"). Linux only, untested since the
# switch to maturin build + install_native.py.
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

# Both targets install the local native extension without letting uv's
# auto-sync reinstall the locked PyPI wheel -- see _select_files in
# scripts/install_native.py for the dist-info rationale. The FIRST uv run
# must NOT use --no-sync (it syncs the locked PyPI chunkhound-native whose
# dist-info install_native.py preserves); every later uv run needs --no-sync.
# A stale 0.1.0 dist-info left by an older 'maturin develop' run is auto-
# corrected: the first sync replaces it with the locked PyPI install.
dev:
	rm -rf target/wheels/
	# Clean ensures a fresh relink even when cargo's incremental cache is stale.
	cargo clean -p chunkhound_native
	$(RUST_DUCKDB_ENV) uv run maturin build --out target/wheels/
	uv run --no-sync python scripts/install_native.py
	uv run --no-sync python scripts/copy_duckdb_runtime.py
	uv run --no-sync pytest tests/test_smoke.py -v -n auto

dev-release:
	rm -rf target/wheels/
	# maturin's wheel-repair step renames the linked libduckdb.so to a hashed
	# name IN PLACE inside cargo's own cached release artifact; a rerun that
	# reuses that cache (no source changes) then fails looking for a hashed
	# name from the previous wheel. Force a clean relink every time.
	cargo clean -p chunkhound_native --release
	$(RUST_DUCKDB_ENV) uv run maturin build --release --out target/wheels/
	uv run --no-sync python scripts/install_native.py
	uv run --no-sync python scripts/copy_duckdb_runtime.py
	uv run --no-sync pytest tests/test_smoke.py -v -n auto

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
