# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepSeek-V4 ROCm gfx950 compressor kernels."""

import statistics
from dataclasses import dataclass

import torch
from tabulate import tabulate

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

KV_BLOCK_SIZE = 16
RMS_EPS = 1e-6
SEED = 2026


@dataclass(frozen=True)
class ShapeConfig:
    name: str
    head_dim: int
    rope_head_dim: int
    ratio: int
    overlap: bool
    state_block_size: int
    quant_format: str

    @property
    def state_width(self) -> int:
        return (2 if self.overlap else 1) * self.head_dim

    @property
    def coff(self) -> int:
        return 2 if self.overlap else 1

    @property
    def token_stride(self) -> int:
        if self.quant_format == "indexer_fp8":
            return self.head_dim
        if self.quant_format == "indexer_mxfp4":
            return self.head_dim // 2
        return (self.head_dim - self.rope_head_dim) + self.rope_head_dim * 2

    @property
    def scale_dim(self) -> int:
        if self.quant_format == "indexer_fp8":
            return 4
        if self.quant_format == "indexer_mxfp4":
            return self.head_dim // 32
        return (self.head_dim - self.rope_head_dim) // 64 + 1

    @property
    def quant_block(self) -> int:
        if self.quant_format == "indexer_fp8":
            return self.head_dim
        if self.quant_format == "indexer_mxfp4":
            return 32
        return 64


SHAPES = {
    "csa": ShapeConfig("csa_main", 512, 64, 4, True, 4, "csa"),
    "hca": ShapeConfig("hca_main", 512, 64, 128, False, 8, "hca"),
    "indexer_fp8": ShapeConfig("indexer_fp8", 128, 64, 4, True, 4, "indexer_fp8"),
    "indexer_mxfp4": ShapeConfig(
        "indexer_mxfp4", 128, 64, 4, True, 4, "indexer_mxfp4"
    ),
}

DEFAULT_SCENARIOS = [
    "decode_boundary",
    "prefill_256",
    "prefill_1024",
    "prefill_4096",
    "prefill_32768",
]


class KVCacheMetadata:
    def __init__(self, slot_mapping: torch.Tensor):
        self.slot_mapping = slot_mapping


@dataclass
class BenchmarkInput:
    name: str
    shape: ShapeConfig
    positions: torch.Tensor
    token_to_req_indices: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    kv_slot_mapping: torch.Tensor
    state_cache_fp32: torch.Tensor
    state_cache_bf16: torch.Tensor
    ape: torch.Tensor
    cos_sin_cache: torch.Tensor
    rms_weight: torch.Tensor
    num_tokens: int
    num_compressed_tokens: int
    num_kv_blocks: int
    kv_block_bytes: int

    def new_kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        flat = torch.empty(
            self.num_kv_blocks,
            self.kv_block_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        flat.zero_()
        return flat, flat.view(self.num_kv_blocks, KV_BLOCK_SIZE, -1)

    def common_kwargs(self, kv_cache: torch.Tensor) -> dict:
        shape = self.shape
        return dict(
            num_actual=self.num_tokens,
            token_to_req_indices=self.token_to_req_indices,
            positions=self.positions,
            slot_mapping=self.slot_mapping,
            block_table=self.block_table,
            block_size=shape.state_block_size,
            state_width=shape.state_width,
            cos_sin_cache=self.cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=KVCacheMetadata(self.kv_slot_mapping),
            pdl_kwargs={},
            head_dim=shape.head_dim,
            rope_head_dim=shape.rope_head_dim,
            compress_ratio=shape.ratio,
            overlap=shape.overlap,
            use_fp4_cache=shape.quant_format == "indexer_mxfp4",
            rms_norm_weight=self.rms_weight,
            rms_norm_eps=RMS_EPS,
            quant_block=shape.quant_block,
            token_stride=shape.token_stride,
            scale_dim=shape.scale_dim,
        )


def _compressed_slot_mapping(positions: list[int], ratio: int) -> list[int]:
    kv_slot_mapping = []
    compressed_idx = 0
    for position in positions:
        if (position + 1) % ratio == 0:
            kv_slot_mapping.append(compressed_idx)
            compressed_idx += 1
        else:
            kv_slot_mapping.append(-1)
    return kv_slot_mapping


def _prefill_scenario(seqlen: int) -> tuple[str, list[int], list[int], list[int]]:
    positions = list(range(seqlen))
    return f"prefill_{seqlen}", positions, [0] * seqlen, positions


def _decode_boundary_scenario(
    shape: ShapeConfig,
) -> tuple[str, list[int], list[int], list[int]]:
    # Boundary positions for both ratio=4 and ratio=128. Include the look-back
    # tokens in the same request so HCA/CSA state reads are fully populated.
    boundary_positions = [127, 255, 383, 511]
    positions, req_indices, slots = [], [], []
    context_len = 4096
    for req_idx, max_position in enumerate(boundary_positions):
        for position in range(max_position + 1):
            positions.append(position)
            req_indices.append(req_idx)
            slots.append(req_idx * context_len + position)
    return "decode_boundary", positions, req_indices, slots


def _scenario(
    shape: ShapeConfig,
    name: str,
) -> tuple[str, list[int], list[int], list[int]]:
    if name == "decode_boundary":
        return _decode_boundary_scenario(shape)
    if name.startswith("prefill_"):
        return _prefill_scenario(int(name.removeprefix("prefill_")))
    available = ["decode_boundary", *[f"prefill_{n}" for n in [256, 1024, 4096, 32768]]]
    raise ValueError(f"Unknown scenario {name!r}. Available scenarios: {available}")


def _build_block_table(
    positions: list[int],
    token_to_req: list[int],
    slot_mapping: list[int],
    state_block_size: int,
) -> torch.Tensor:
    num_reqs = max(token_to_req) + 1 if token_to_req else 1
    max_logical_block = max(positions) // state_block_size + 2
    block_table = torch.zeros(
        num_reqs,
        max_logical_block,
        dtype=torch.int32,
        device="cuda",
    )
    for req_idx, position, slot in zip(token_to_req, positions, slot_mapping):
        block_table[req_idx, position // state_block_size] = slot // state_block_size
    return block_table


def _cos_sin_cache(max_position: int, rope_head_dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, rope_head_dim, 2, dtype=torch.float32, device="cuda")
            / rope_head_dim
        )
    )
    freqs = torch.outer(
        torch.arange(max_position + 1, dtype=torch.float32, device="cuda"),
        inv_freq,
    )
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def build_input(shape: ShapeConfig, scenario_name: str) -> BenchmarkInput:
    name, positions, token_to_req, slot_mapping = _scenario(shape, scenario_name)
    num_tokens = len(positions)
    kv_slot_mapping = _compressed_slot_mapping(positions, shape.ratio)
    num_compressed_tokens = sum(slot >= 0 for slot in kv_slot_mapping)

    generator = torch.Generator(device="cuda").manual_seed(SEED + num_tokens)
    kv = torch.randn(
        num_tokens,
        shape.coff * shape.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    score = torch.randn_like(kv)
    ape = torch.randn(
        shape.ratio,
        shape.coff * shape.head_dim,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    rms_weight = torch.rand(
        shape.head_dim,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).to(torch.bfloat16)

    positions_t = torch.tensor(positions, dtype=torch.int64, device="cuda")
    token_to_req_t = torch.tensor(token_to_req, dtype=torch.int32, device="cuda")
    slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")
    kv_slot_mapping_t = torch.tensor(kv_slot_mapping, dtype=torch.int64, device="cuda")

    max_slot = max(slot_mapping) if slot_mapping else 0
    num_state_blocks = max_slot // shape.state_block_size + 4
    state_shape = (num_state_blocks, shape.state_block_size, 2 * shape.state_width)
    state_cache_fp32 = torch.zeros(state_shape, dtype=torch.float32, device="cuda")
    state_cache_bf16 = torch.zeros(state_shape, dtype=torch.bfloat16, device="cuda")

    save_partial_states(
        kv=kv,
        score=score,
        ape=ape,
        positions=positions_t,
        state_cache=state_cache_fp32,
        slot_mapping=slot_mapping_t,
        block_size=shape.state_block_size,
        state_width=shape.state_width,
        compress_ratio=shape.ratio,
    )
    save_partial_states(
        kv=kv,
        score=score,
        ape=None,
        positions=positions_t,
        state_cache=state_cache_bf16,
        slot_mapping=slot_mapping_t,
        block_size=shape.state_block_size,
        state_width=shape.state_width,
        compress_ratio=shape.ratio,
    )

    num_kv_blocks = num_compressed_tokens // KV_BLOCK_SIZE + 4
    kv_block_bytes = KV_BLOCK_SIZE * (shape.token_stride + shape.scale_dim)
    return BenchmarkInput(
        name=name,
        shape=shape,
        positions=positions_t,
        token_to_req_indices=token_to_req_t,
        slot_mapping=slot_mapping_t,
        block_table=_build_block_table(
            positions, token_to_req, slot_mapping, shape.state_block_size
        ),
        kv_slot_mapping=kv_slot_mapping_t,
        state_cache_fp32=state_cache_fp32,
        state_cache_bf16=state_cache_bf16,
        ape=ape,
        cos_sin_cache=_cos_sin_cache(max(positions), shape.rope_head_dim),
        rms_weight=rms_weight,
        num_tokens=num_tokens,
        num_compressed_tokens=num_compressed_tokens,
        num_kv_blocks=num_kv_blocks,
        kv_block_bytes=kv_block_bytes,
    )


def hip_available() -> bool:
    try:
        import vllm._rocm_C  # noqa: F401

        return hasattr(torch.ops._rocm_C, "dsv4_csa_compress")
    except Exception:
        return False


def run_triton(inputs: BenchmarkInput, kv_cache: torch.Tensor) -> None:
    compress_norm_rope_store_triton(
        state_cache=inputs.state_cache_fp32,
        **inputs.common_kwargs(kv_cache),
    )


def run_hip(inputs: BenchmarkInput, kv_cache: torch.Tensor) -> None:
    shape = inputs.shape
    common = (
        inputs.state_cache_bf16,
        inputs.num_tokens,
        inputs.ape,
        inputs.token_to_req_indices,
        inputs.positions,
        inputs.slot_mapping,
        inputs.block_table,
        shape.state_block_size,
        inputs.rms_weight.to(torch.float32),
        RMS_EPS,
        inputs.cos_sin_cache,
        kv_cache,
        inputs.kv_slot_mapping,
        kv_cache.shape[1],
        shape.scale_dim,
    )
    if shape.head_dim == 128:
        torch.ops._rocm_C.dsv4_indexer_compress(
            *common, shape.quant_format == "indexer_mxfp4"
        )
    elif shape.ratio == 128:
        plan_capacity = (
            inputs.num_tokens // shape.ratio + inputs.block_table.shape[0] + 2
        )
        plan_scratch = torch.empty(plan_capacity, dtype=torch.int32, device="cuda")
        counter_scratch = torch.empty(1, dtype=torch.int32, device="cuda")
        torch.ops._rocm_C.dsv4_hca_compress(
            *common,
            plan_scratch,
            counter_scratch,
        )
    else:
        torch.ops._rocm_C.dsv4_csa_compress(*common)


def graph_replay_us(fn, launches: int, reps: int, warmup: int) -> float:
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        for _ in range(launches):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / launches)
    return statistics.median(samples)


def median_min_max(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), min(values), max(values)


def format_range(values: tuple[float, float, float]) -> str:
    median, low, high = values
    return f"{median:.2f}[{low:.2f}-{high:.2f}]"


def benchmark_scenario(
    inputs: BenchmarkInput,
    rounds: int,
    launches: int,
    reps: int,
    warmup: int,
    have_triton: bool,
) -> dict:
    _, kv_cache = inputs.new_kv_cache()
    hip_fn = lambda: run_hip(inputs, kv_cache)
    triton_fn = (lambda: run_triton(inputs, kv_cache)) if have_triton else None

    for _ in range(30):
        if triton_fn is not None:
            triton_fn()
        hip_fn()
    torch.cuda.synchronize()

    hip_times, triton_times, ratios = [], [], []
    for _ in range(rounds):
        hip_us = graph_replay_us(hip_fn, launches, reps, warmup)
        hip_times.append(hip_us)
        if triton_fn is not None:
            triton_us = graph_replay_us(triton_fn, launches, reps, warmup)
            triton_times.append(triton_us)
            ratios.append(hip_us / triton_us)

    result = {"scenario": inputs.name, "hip": median_min_max(hip_times)}
    if triton_times:
        result["triton"] = median_min_max(triton_times)
        result["ratio"] = median_min_max(ratios)
    return result


def print_results(shape: ShapeConfig, results: list[dict], markdown: bool) -> None:
    if markdown:
        print(f"#### {shape.name}")
        print()
        print(
            "| Scenario | Triton us/launch | HIP us/launch | "
            "HIP/Triton median[min..max] |"
        )
        print("|---|---:|---:|---:|")
        for result in results:
            triton = f"{result['triton'][0]:.1f}" if "triton" in result else "n/a"
            ratio = format_range(result["ratio"]) if "ratio" in result else "n/a"
            print(
                f"| {result['scenario']} | {triton} | "
                f"{result['hip'][0]:.1f} | {ratio} |"
            )
    else:
        rows = []
        for result in results:
            rows.append(
                [
                    result["scenario"],
                    f"{result['triton'][0]:.1f}" if "triton" in result else "n/a",
                    f"{result['hip'][0]:.1f}",
                    format_range(result["ratio"]) if "ratio" in result else "n/a",
                ]
            )
        print(f"DeepSeek-V4 compressor benchmark ({shape.name})")
        print("HIP/Triton < 1.0 means HIP is faster. Times are us/launch.")
        print(
            tabulate(
                rows,
                headers=["scenario", "triton", "hip", "hip/triton"],
                tablefmt="github",
            )
        )

    ratios = [result["ratio"][0] for result in results if "ratio" in result]
    if ratios:
        print()
        print(f"Geomean HIP/Triton: {statistics.geometric_mean(ratios):.3f}")


def parse_args():
    parser = FlexibleArgumentParser(
        description="Benchmark DeepSeek-V4 ROCm gfx950 compressor kernels."
    )
    parser.add_argument("--shape", choices=sorted(SHAPES), default="csa")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--launches", type=int, default=100)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is not available")
    if not hip_available():
        raise SystemExit("dsv4 compressor ops are not registered in _rocm_C")

    shape = SHAPES[args.shape]
    have_triton = shape.quant_format != "indexer_mxfp4"
    results = []
    for scenario_name in args.scenarios:
        inputs = build_input(shape, scenario_name)
        results.append(
            benchmark_scenario(
                inputs,
                rounds=args.rounds,
                launches=args.launches,
                reps=args.reps,
                warmup=args.warmup,
                have_triton=have_triton,
            )
        )

    print_results(shape, results, markdown=args.markdown)


if __name__ == "__main__":
    main()
