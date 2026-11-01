.PHONY: install run shell clean modal-setup modal-secret modal-deploy modal-prefetch help

.DEFAULT_GOAL := help

HOST ?= 127.0.0.1
PORT ?= 8000

help:
	@echo "3DViewer — Poetry-managed Python env"
	@echo ""
	@echo "  install          - Install dependencies (and build torchmcubes if needed)"
	@echo "  run              - Start FastAPI + static viewer (http://$(HOST):$(PORT))"
	@echo "  shell            - Open a Poetry-managed shell"
	@echo "  modal-setup      - Authenticate the Modal CLI (opens a browser)"
	@echo "  modal-secret     - Create Modal secret huggingface-secret from HF_TOKEN in .env"
	@echo "  modal-deploy     - Deploy cloud/modal_app.py (Hunyuan3D-2mv on A10G)"
	@echo "  modal-prefetch   - Download Hunyuan weights into the Modal HF volume"
	@echo "  clean            - Remove Poetry env and caches (not vendor/ or models)"
	@echo ""
	@echo "InstantMesh stays on venv-instantmesh/ (separate stack)."

install:
	@if [ ! -x .venv/bin/python ]; then \
		PYTHON_BIN="$$(command -v python3.11 || true)"; \
		if [ -z "$$PYTHON_BIN" ]; then echo "Error: python3.11 not found on PATH." >&2; exit 1; fi; \
		echo "==> Creating .venv with $$PYTHON_BIN"; \
		"$$PYTHON_BIN" -m venv .venv; \
	fi
	poetry install
	@poetry run python -c "import torchmcubes" 2>/dev/null || ( \
		echo "==> Building torchmcubes against installed PyTorch..." && \
		poetry run pip install scikit-build-core cmake ninja pybind11 && \
		poetry run pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git \
	)

run:
	PYTORCH_ENABLE_MPS_FALLBACK=1 poetry run uvicorn backend.app.main:app --host $(HOST) --port $(PORT) --reload

modal-setup:
	poetry run python -m modal setup

# Reads HF_TOKEN from the environment or .env; the token is not printed.
modal-secret:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	if [ -z "$$HF_TOKEN" ]; then echo "Error: HF_TOKEN is empty. Put it in .env first." >&2; exit 1; fi; \
	poetry run modal secret create huggingface-secret HF_TOKEN="$$HF_TOKEN"

modal-deploy:
	poetry run modal deploy cloud/modal_app.py

modal-prefetch:
	poetry run modal run cloud/modal_app.py::prefetch

shell:
	poetry run $(SHELL)

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
