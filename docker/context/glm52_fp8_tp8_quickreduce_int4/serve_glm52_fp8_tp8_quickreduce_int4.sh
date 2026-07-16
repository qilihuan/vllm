#!/bin/bash
set -euo pipefail

export HF_HOME=/data/huggingface
export HF_HUB_CACHE=/data/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/data/huggingface/hub
export HF_XET_HIGH_PERFORMANCE=1

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_IPC_MODE_LEGACY=1
export TOKENIZERS_PARALLELISM=false

export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16="${VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:-1}"
export VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB="${VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB:-32}"
export VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB="${VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB:-512}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-glm52_fp8_tp8_quickreduce_int4_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-/workspace/runs}"
export RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${EXPERIMENT_NAME}}"
export TUNE_DIR="${TUNE_DIR:-/opt/glm52/tuning/final_gemm}"
export A8W8_TUNE="${A8W8_TUNE:-${TUNE_DIR}/glm52_chunk16k_final_a8w8.csv}"
export BF16_TUNE="${BF16_TUNE:-${TUNE_DIR}/glm52_chunk16k_final_bf16.csv}"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE:-/opt/glm52/tuning/runtime/a8w8_blockscale_tuned_gemm.runtime_dedup.csv}"
export AITER_CONFIG_GEMM_BF16="${AITER_CONFIG_GEMM_BF16:-/opt/glm52/tuning/runtime/bf16_tuned_gemm.runtime_dedup.csv}"
export MODEL="${MODEL:-/data/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541}"
export SPECULATIVE_CONFIG_JSON="${SPECULATIVE_CONFIG_JSON:-}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-66000}"

mkdir -p "${RUN_DIR}"
sha256sum "${A8W8_TUNE}" "${BF16_TUNE}" | tee "${RUN_DIR}/embedded_gemm.sha256"

if [[ ! -f "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}" || ! -f "${AITER_CONFIG_GEMM_BF16}" ]]; then
python3 - <<'PY'
import os
from pathlib import Path

import pandas as pd

run_dir = Path(os.environ["RUN_DIR"])
tune_dir = Path(os.environ["TUNE_DIR"])
config_dir = Path("/usr/local/lib/python3.12/dist-packages/aiter/configs")


def build_runtime_config(tuned_name: str, extra_csv: Path, key_cols: list[str]) -> Path:
    paths = [config_dir / f"{tuned_name}.csv"]
    model_config_dir = config_dir / "model_configs"
    if model_config_dir.exists():
        paths.extend(
            sorted(
                p
                for p in model_config_dir.glob(f"*{tuned_name}*.csv")
                if p.is_file() and "untuned" not in p.name
            )
        )
    paths.append(extra_csv)

    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "gfx" in df.columns:
            df = df[df["gfx"] != "gfx"]
        for col in ["cu_num", "M", "N", "K"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.dropna(
        subset=[c for c in ["cu_num", "M", "N", "K"] if c in merged.columns]
    )
    for col in ["cu_num", "M", "N", "K"]:
        if col in merged.columns:
            merged[col] = merged[col].astype("int64")

    dedup_keys = [col for col in key_cols if col in merged.columns]
    merged = merged.drop_duplicates(subset=dedup_keys, keep="last")

    out = run_dir / f"{tuned_name}.runtime_dedup.csv"
    merged.to_csv(out, index=False)
    print(f"{tuned_name}: wrote {len(merged)} rows to {out}", flush=True)
    return out


a8w8_runtime = build_runtime_config(
    "a8w8_blockscale_tuned_gemm",
    tune_dir / "glm52_chunk16k_final_a8w8.csv",
    ["gfx", "cu_num", "M", "N", "K"],
)
bf16_runtime = build_runtime_config(
    "bf16_tuned_gemm",
    tune_dir / "glm52_chunk16k_final_bf16.csv",
    [
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
    ],
)

with open(run_dir / "runtime_gemm_env.sh", "w", encoding="utf-8") as f:
    f.write(f"export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE={a8w8_runtime}\n")
    f.write(f"export AITER_CONFIG_GEMM_BF16={bf16_runtime}\n")
PY

source "${RUN_DIR}/runtime_gemm_env.sh"
fi

sha256sum "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}" "${AITER_CONFIG_GEMM_BF16}" \
  | tee "${RUN_DIR}/runtime_gemm.sha256"

cat > "${RUN_DIR}/server_command.txt" <<EOF
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}
export AITER_CONFIG_GEMM_BF16=${AITER_CONFIG_GEMM_BF16}
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION}
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=${VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16}
export VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=${VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB}
export VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=${VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB}
vllm serve ${MODEL} --served-model-name glm-5.2-fp8 --host 0.0.0.0 --port 7777 --tensor-parallel-size 8 --distributed-executor-backend mp --no-enable-prefix-caching --gpu-memory-utilization 0.80 --kv-cache-dtype fp8_e4m3 --max-num-seqs 32 --trust-remote-code --moe-backend aiter --linear-backend aiter --chat-template-content-format string --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --max-model-len ${MAX_MODEL_LEN} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --num-gpu-blocks-override ${NUM_GPU_BLOCKS_OVERRIDE}
EOF

if [[ -n "${SPECULATIVE_CONFIG_JSON}" ]]; then
  printf -- " --speculative-config '%s'\n" "${SPECULATIVE_CONFIG_JSON}" >> "${RUN_DIR}/server_command.txt"
fi

declare -a SPEC_ARGS=()
if [[ -n "${SPECULATIVE_CONFIG_JSON}" ]]; then
  SPEC_ARGS=(--speculative-config "${SPECULATIVE_CONFIG_JSON}")
fi

vllm serve "${MODEL}" \
  --served-model-name glm-5.2-fp8 \
  --host 0.0.0.0 --port 7777 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 32 \
  --trust-remote-code \
  --moe-backend aiter \
  --linear-backend aiter \
  --chat-template-content-format string \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
  "${SPEC_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/server.log"
