# vLLM Copilot Instructions

## Activate Environment
Need activate the vLLM development environment when creating a new terminal:
```bash
source /media/minhdt/DATA/Code/test_ocr/training_tiny_llm/dev_vllm/vllm/.venv/bin/activate
```

## Build & Install

**Python-only development (no CUDA changes):**
```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

**Python + CUDA/C++ development:**
```bash
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
uv pip install -e . --no-build-isolation
```

Develop with Python 3.12 to match CI.

## Linting & Formatting

Uses `pre-commit` (ruff + mypy + typos):
```bash
pre-commit install          # set up hooks
pre-commit run              # run on staged files
pre-commit run -a           # run on all files
pre-commit run --hook-stage manual mypy-3.10  # run mypy locally
```

## Testing

```bash
# Install test dependencies
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# Run all tests
pytest tests/

# Run a single test file with output
pytest -s -v tests/test_logger.py
```

Pytest markers: `slow_test`, `core_model`, `distributed`, `cpu_test`, `skip_v1`, `optional`.

## Architecture

Request flow: **Entrypoint → LLMEngine → Worker → ModelRunner → Model**

- **`vllm/entrypoints/`** — `LLM` class (offline inference) and `vllm serve` / OpenAI-compatible API server
- **`vllm/engine/`** — `LLMEngine` (sync) and `AsyncLLMEngine` (async online serving); handles scheduling, input processing, output decoding
- **Worker** — one process per accelerator device; orchestrated via rank/local_rank for TP/PP
- **ModelRunner** — lives inside each worker; loads model, prepares input tensors, manages CUDA graphs
- **`vllm/model_executor/models/`** — 150+ model implementations; each is a `torch.nn.Module`
- **`vllm/config/`** — `VllmConfig` aggregates all sub-configs (`ModelConfig`, `CacheConfig`, `ParallelConfig`, `SchedulerConfig`, `AttentionConfig`, etc.) and is passed through the entire class hierarchy as the single engine-level global state

The V1 engine (`vllm/v1/`) is the next-generation optimized execution path.

## Key Conventions

### Model Constructor Signature

All vLLM models must use this keyword-only signature:
```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    config = vllm_config.model_config.hf_config
    cache_config = vllm_config.cache_config
    quant_config = vllm_config.quant_config
```

The `prefix` argument matches the module's state dict key in the checkpoint (e.g., `""`, `"vision"`, `"language"`).

### Adding a New Model

1. Implement the model in `vllm/model_executor/models/<model_name>.py`
2. Register it in `vllm/model_executor/models/registry.py` under `_VLLM_MODELS` (keep lists alphabetical)
3. Update `docs/models/supported_models.md`
4. For multimodal models, implement the [`SupportsMultiModal`](../vllm/model_executor/models/interfaces.py) interface

Out-of-tree models can be registered via a plugin without modifying vLLM source:
```python
from vllm import ModelRegistry
ModelRegistry.register_model("YourModelForCausalLM", "your_module:YourModelClass")
```

### VllmConfig is the Single Source of Truth

Never add constructor parameters to pass individual config values down the class hierarchy. Add new config fields to `VllmConfig` (in `vllm/config/`) and read them directly where needed.

### Model Capability Interfaces

Use marker interfaces from `vllm/model_executor/models/interfaces.py` to declare model capabilities:
- `SupportsMultiModal` — image/audio/video inputs
- `SupportsLoRA` — LoRA adapter support
- `SupportsPP` — pipeline parallelism

### Multimodal Pipeline

Multimodal models process inputs through: image processor → `embed_modality()` → projector → LM. See `vllm/multimodal/` for input types and `MultiModalFieldConfig`.

### Attention Backends

Pluggable backends (Flash-Attention, FlashInfer, xFormers, Triton) are auto-selected by hardware/precision. Configured via `vllm/config/attention.py`.

## Pull Requests

Commits require a `Signed-off-by:` header (DCO):
```bash
git commit -s -m "message"
```

PR titles must use one of: `[Bugfix]`, `[CI/Build]`, `[Doc]`, `[Model]`, `[Frontend]`, `[Kernel]`, `[Core]`, `[RFC]`.

## Custom Model in This Fork

This fork includes `NanoVLM` — a custom vision-language model:
- Implementation: `vllm/model_executor/models/nano_vlm.py`
- Submodules: `vllm/model_executor/models/nano_vlm_modules/`
- Registered in `vllm/model_executor/models/registry.py`
- Implements `SupportsMultiModal`
