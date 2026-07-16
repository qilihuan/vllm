# GLM-5.2-FP8 ROCm Image Recipe

This directory builds a reusable ROCm image for `GLM-5.2-FP8` with:

- final GEMM tune embedded in the image
- QuickReduce `INT4` defaults
- `TP8`
- chunk `16k`
- default `--max-model-len 131072`

Container-side paths are neutral and do not depend on any user-specific host
path:

- tune dir: `/opt/glm52/tuning/final_gemm`
- runtime tuned configs: `/opt/glm52/tuning/runtime`
- entrypoint: `/opt/glm52/bin/serve_glm52_fp8_tp8_quickreduce_int4.sh`
- run dir: `/workspace/runs`

## Build

From the repo root:

```bash
docker build \
  -t glm52-fp8-tp8-quickreduce-int4:final-gemm \
  -f docker/context/glm52_fp8_tp8_quickreduce_int4/Dockerfile \
  docker/context/glm52_fp8_tp8_quickreduce_int4
```

## Run via Entrypoint

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --ipc host --network host --cap-add SYS_PTRACE \
  --shm-size 32g --pids-limit 0 \
  --ulimit nproc=4194304:4194304 --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  -e MAX_MODEL_LEN=131072 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -v /data:/data \
  -v /mnt:/mnt \
  -v /tmp/glm52_runs:/workspace/runs \
  glm52-fp8-tp8-quickreduce-int4:final-gemm
```

## Run via `vllm serve`

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --ipc host --network host --cap-add SYS_PTRACE \
  --shm-size 32g --pids-limit 0 \
  --ulimit nproc=4194304:4194304 --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HSA_ENABLE_IPC_MODE_LEGACY=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=/opt/glm52/tuning/runtime/a8w8_blockscale_tuned_gemm.runtime_dedup.csv \
  -e AITER_CONFIG_GEMM_BF16=/opt/glm52/tuning/runtime/bf16_tuned_gemm.runtime_dedup.csv \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1 \
  -e VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=32 \
  -e VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512 \
  -v /data:/data \
  -v /mnt:/mnt \
  --entrypoint bash \
  glm52-fp8-tp8-quickreduce-int4:final-gemm \
  -lc 'vllm serve /data/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541 --served-model-name glm-5.2-fp8 --host 0.0.0.0 --port 7777 --tensor-parallel-size 8 --distributed-executor-backend mp --no-enable-prefix-caching --gpu-memory-utilization 0.80 --kv-cache-dtype fp8_e4m3 --max-num-seqs 32 --trust-remote-code --moe-backend aiter --linear-backend aiter --chat-template-content-format string --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --max-model-len 131072 --max-num-batched-tokens 16384 --num-gpu-blocks-override 66000'
```

## MTP Variants

Set `SPECULATIVE_CONFIG_JSON` at runtime, for example:

```bash
-e SPECULATIVE_CONFIG_JSON='{"method":"mtp","num_speculative_tokens":3}'
```
