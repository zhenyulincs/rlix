# Vast.ai Setup Guide — MILES RLix Integration Tests

## Hardware

| Field | Value |
|-------|-------|
| Instance ID | 35236058 |
| GPU | 4× NVIDIA RTX A5000 (24 GB each = 96 GB total) |
| VRAM per GPU | 24,564 MiB |
| RAM | 512 GB |
| CPU | 24 cores |
| CUDA | 12.6 |
| Driver | 535.54.03 |
| SSH | `ssh -i ~/.ssh/general_private_key -p 45678 root@ssh9.vast.ai` (check `vastai ssh-url <id>` after each start) |

## One-Time Environment Setup

```bash
# Activate working Python venv (has torch 2.9.1+cu128)
source /root/rlix/.venv/bin/activate

# Get SSH connection info
vastai ssh-url 35236058
```

### Step 1 — Clone rlix and initialize submodules
```bash
cd /workspace
git clone https://github.com/zhenyulincs/rlix.git rlix_test
cd rlix_test
git checkout task11-miles-rlix

# Miles submodule (rlix-task11 branch)
git config submodule.external/miles.branch rlix-task11
git submodule update --init --remote external/miles

# NeMo submodule (for NeMo pipeline tests)
git submodule update --init external/NeMo

# ROLL submodule (rlix branch)
git clone https://github.com/rlops/ROLL.git --branch rlix --depth=1 external/ROLL
```

### Step 2 — Install Python dependencies
```bash
# All installs use /root/rlix/.venv (has torch + megatron-core + sglang ready)
VENV=/root/rlix/.venv/bin

# Core packages
$VENV/pip3 install ray[default] -q
$VENV/pip3 install sglang[all] \
  --extra-index-url https://flashinfer.ai/whl/cu124/torch2.4/ -q
$VENV/pip3 install sglang-router wandb tensorboard codetiming psutil -q
$VENV/pip3 install 'numpy<2.0.0' -q

# Megatron-LM (provides megatron.training for MILES)
cd /workspace
git clone https://github.com/radixark/Megatron-LM.git --branch miles-main --depth=1
cd Megatron-LM && $VENV/pip3 install -e . --no-deps -q

# rlix + submodules
cd /workspace/rlix_test
$VENV/pip3 install -e external/ROLL --no-deps -q
$VENV/pip3 install -e . --no-deps -q
$VENV/pip3 install -e external/miles --no-deps -q
```

### Step 3 — Download test model
```bash
/root/rlix/.venv/bin/python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
tok.save_pretrained('/workspace/models/Qwen2.5-0.5B')
model.save_pretrained('/workspace/models/Qwen2.5-0.5B')
"
```

### Step 4 — Start Ray cluster
```bash
/root/rlix/.venv/bin/ray start --head --num-gpus=4
```

---

## Running Tests

### Unit Tests (no GPU needed)
```bash
cd /workspace/rlix_test
export PYTHONPATH=/workspace/rlix_test:/workspace/rlix_test/external/miles:/workspace/rlix_test/external/ROLL:/workspace/Megatron-LM
/root/rlix/.venv/bin/python -m pytest tests/ --ignore=tests/integration -q
```
**Expected: 148 pass, 1 fail (test_vllm_backend_receiver — needs torch/ray in same env)**

### Gate Tests (GPU required)
Set `export PYTHONPATH=/workspace/rlix_test:/workspace/rlix_test/external/miles:/workspace/rlix_test/external/ROLL:/workspace/Megatron-LM`

**Gate 1 — Sleep/wake + router admission**
```bash
/root/rlix/.venv/bin/python -c "
# Tests EngineInfo state machine, sleep_partial/wake_partial, _use_url routing
from miles.ray.rollout import EngineInfo
# ... (see docs/TASK11_FB_IMPL.md for test script)
"
```
Pass conditions: EngineInfo transitions active→disabling→offloaded→loading→active; router skips disabled workers

**Gate 2 — F10 topology validation**
```bash
/root/rlix/.venv/bin/python -c "
import importlib.util, sys
sys.path.insert(0, '/workspace/rlix_test')
spec = importlib.util.spec_from_file_location('mp', '/workspace/rlix_test/rlix/pipeline/miles_pipeline.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from argparse import Namespace
# 1-engine topology should be rejected
args = Namespace(actor_num_nodes=1, actor_num_gpus_per_node=2, rollout_num_gpus=2,
    rollout_num_gpus_per_engine=2, offload_train=True, model_update_transport='cpu_serialize',
    sglang_data_parallel_size=1, expert_model_parallel_size=1, moe_router_topk=0,
    async_save=False, rollout_force_stream=False, miles_router_middleware_paths=[],
    critic_model_path=None, reward_model_path=None, sglang_secondary_server_count=0,
    tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
    num_gpus_per_node=2, miles_model_update_bucket_size_mb=512,
    rollout_function_path='examples.fully_async.fully_async_rollout.generate_rollout_fully_async')
try: mod._validate_rlix_topology(args); print('FAIL')
except AssertionError as e: print('PASS:', str(e)[:60])
"
```

**Gate 2.5 — CPU bucket cache + wire format**
```bash
/root/rlix/.venv/bin/python -c "
from miles.backends.megatron_utils.update_weight.cpu_bucket_cache import CpuBucketCache, _pack_tensors_to_buckets
import torch, io
tensors = [('layer.weight', torch.randn(512, 512)) for _ in range(4)]
records = _pack_tensors_to_buckets(tensors, 4*1024*1024)
cache = CpuBucketCache()
cache.build(tensors, step=3, bucket_size_bytes=4*1024*1024, check_host_ram=False)
payload = cache.serialize_bucket_to_bytes(0)
data = torch.load(io.BytesIO(payload), weights_only=False)
assert 'bucket' in data and 'tensors_meta' in data
print('PASS: bucket cache build + serialize OK, payload size:', len(payload))
"
```

**Gate 3 — SGLang GPU loading (Qwen 0.5B, tp=2, GPU 0+1)**

⚠️ First-run caveat: Triton JIT compilation takes 15-30 minutes on a fresh instance. On subsequent runs kernels are cached (~2 min startup).

```bash
# Wait for health endpoint to confirm server ready
python /workspace/sglang_gpu_test.py
```
Expected during load: `GPU0:100%util/409MiB  GPU1:100%util/410MiB`
Expected after ready: server responds on port 30000, inference works

---

## Resource Utilization Summary

| Phase | CPU | RAM | GPU0 | GPU1 | GPU2 | GPU3 |
|-------|-----|-----|------|------|------|------|
| Baseline | 0.5% | 14 GB | 0%/18 MiB | 0%/18 MiB | 0%/18 MiB | 0%/18 MiB |
| Kernel JIT (loading) | 7% | 18 GB | **100%/409 MiB** | **100%/410 MiB** | 0%/18 MiB | 0%/18 MiB |
| Model ready (idle) | <1% | 18 GB | 5%/~3 GB | 5%/~3 GB | 0%/18 MiB | 0%/18 MiB |
| Inference (generating) | 10% | 18 GB | **95%/~12 GB** | **95%/~12 GB** | 0%/18 MiB | 0%/18 MiB |

Note: Gate tests confirmed Qwen 0.5B loads successfully with 409 MiB/GPU (model weights in bf16, tp=2). Full KV cache allocation brings GPU memory to ~12 GB during inference.

---

## Known Issues

1. **Triton JIT first-run**: SGLang compiles attention kernels on first boot (15-30 min). Subsequent runs use cache (~2 min). MILES Docker pre-compiles kernels — this issue is specific to bare host installs.

2. **megatron.bridge missing**: `miles_plugins.megatron_bridge` warning is benign — the bridge shim is for optional performance features, not required for correctness.

3. **transformer_engine**: Not installable on cu126 system (needs cu128). Megatron-core falls back to torch-native implementations — warnings only, no correctness impact.

4. **Full MILES arg parser**: MILES `parse_args()` integrates with Megatron-LM's argparser which requires many flags. Use the MILES Docker for full `train_async.py` runs. The RLix control plane (`run_miles_rlix.py`) provides its own init path that bypasses the standalone trainer.
