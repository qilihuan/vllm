FROM docker.io/vllm/vllm-openai-rocm@sha256:123c73cb8328230026cc3758425df0134dea15a6a0f25d49728f85c570daec09

WORKDIR /opt/glm52

COPY serve_glm52_fp8_tp8_quickreduce_int4.sh /opt/glm52/bin/serve_glm52_fp8_tp8_quickreduce_int4.sh
COPY glm52_chunk16k_final_a8w8.csv /opt/glm52/tuning/final_gemm/glm52_chunk16k_final_a8w8.csv
COPY glm52_chunk16k_final_bf16.csv /opt/glm52/tuning/final_gemm/glm52_chunk16k_final_bf16.csv

RUN chmod 0755 /opt/glm52/bin/serve_glm52_fp8_tp8_quickreduce_int4.sh && \
    mkdir -p /opt/glm52/tuning/runtime && \
    python3 - <<'PY'
from pathlib import Path

import pandas as pd

config_dir = Path("/usr/local/lib/python3.12/dist-packages/aiter/configs")
tune_dir = Path("/opt/glm52/tuning/final_gemm")
runtime_dir = Path("/opt/glm52/tuning/runtime")


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

    out = runtime_dir / f"{tuned_name}.runtime_dedup.csv"
    merged.to_csv(out, index=False)
    print(f"{tuned_name}: wrote {len(merged)} rows to {out}", flush=True)
    return out


build_runtime_config(
    "a8w8_blockscale_tuned_gemm",
    tune_dir / "glm52_chunk16k_final_a8w8.csv",
    ["gfx", "cu_num", "M", "N", "K"],
)
build_runtime_config(
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
PY

ENV TUNE_DIR=/opt/glm52/tuning/final_gemm
ENV A8W8_TUNE=/opt/glm52/tuning/final_gemm/glm52_chunk16k_final_a8w8.csv
ENV BF16_TUNE=/opt/glm52/tuning/final_gemm/glm52_chunk16k_final_bf16.csv
ENV AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=/opt/glm52/tuning/runtime/a8w8_blockscale_tuned_gemm.runtime_dedup.csv
ENV AITER_CONFIG_GEMM_BF16=/opt/glm52/tuning/runtime/bf16_tuned_gemm.runtime_dedup.csv
ENV RUN_ROOT=/workspace/runs
ENV MAX_MODEL_LEN=131072
ENV VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
ENV VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1
ENV VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=32
ENV VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512

ENTRYPOINT ["/opt/glm52/bin/serve_glm52_fp8_tp8_quickreduce_int4.sh"]
