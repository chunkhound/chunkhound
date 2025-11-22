# Evaluation helpers (sane defaults, two targets):
#   make bench-lang      # semantic language eval on lang-eval-dev @ k=10
#   make bench-cluster   # clustering eval on cluster-stress-dev
#
# Optional: override config file via CONFIG, e.g.:
#   make bench-lang CONFIG=.chunkhound.json

.PHONY: bench-lang bench-cluster

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
