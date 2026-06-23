# AMD DeepSeek-V4 fused compressors (gfx950 / CDNA4)

The DeepSeek-V4 compressors on AMD gfx950 (CDNA4): single fused, plan-free HIP
kernels producing the **same vLLM cache format** as the production Triton FP32
kernel (NOPE FP8 + ROPE bf16 + UE8M0 scales; indexer adds an MXFP4 variant),
validated **byte-exact** against it.

Three shapes share one design (compress → softmax → weighted-sum → RMSNorm →
RoPE → quant, all in-register, `warp_reduce` RMSNorm, no LDS/`__syncthreads`):

| op | head_dim | ratio | notes |
|----|----------|-------|-------|
| `dsv4_csa_compress`     | 512 | 4   | overlap, K_POOL=8 |
| `dsv4_hca_compress`     | 512 | 128 | no overlap, K-split + compact plan |
| `dsv4_indexer_compress` | 128 | 4   | FP8 or MXFP4 (`use_fp4_cache`) |

## Build & wiring (AOT)

The kernels live in `csrc/rocm/dsv4_{csa,hca,indexer}_compress.cu` and are built
into the `_rocm_C` extension as `torch.ops._rocm_C.dsv4_*` **only when gfx950 is
among `VLLM_GPU_ARCHES`** (CMake `VLLM_ROCM_GFX950`). The sources are multi-arch
safe: the device body is real on the gfx950 pass and an empty stub elsewhere
(the native FP8/FP4 cvt builtins are CDNA4-only).

The opt-in path routes through `compress_norm_rope_store_hip` in
`hip_compress_dispatch.py`, enabled via
**`VLLM_ROCM_DSV4_HIP_COMPRESSOR=1`** (requires
`VLLM_ROCM_DSV4_BF16_STATE_CACHE=1` for HIP-selected compressors). `1`/`true` enables
only CSA+HCA by default, keeping indexer on Triton; use
`VLLM_ROCM_DSV4_HIP_COMPRESSOR=csa,hca,indexer` to explicitly include HIP indexer.
`hip_compressor_supported(...)` gates on the selected mode + gfx950 + bf16 state
cache + the uint8 paged layout; otherwise the model falls back to Triton (or
raises a clear error when bf16 state cache is on for a HIP-selected compressor,
since Triton has no bf16-state-cache path).

## Correctness

Byte-exact vs Triton FP32 on every scenario; bit-faithful quantizer (FP8 E4M3
RNE, bf16 RoPE, UE8M0 `ceil(log2)`). The indexer MXFP4 tail matches a faithful
torch front-end + RNE reference (no AMD Triton MXFP4 oracle exists).

## Performance

HIP speedup over the vLLM Triton FP32 kernel (**×, = triton / hip**; >1 = HIP
faster), kernel device-time, gfx950 (MI355). Measured with
`benchmark_dsv4_compress.py` (HIP-graph replay, same-process A/B ratio — see the
measurement note below; absolute µs are not comparable across machines/clocks).

| scenario | CSA (512, 4) | HCA (512, 128) | indexer FP8 (128, 4) |
|----------|:------------:|:--------------:|:--------------------:|
| decode (1 tok/req) | 1.2× | 4.4× | 0.6× |
| prefill 1k         | 1.1× | 2.2× | 0.5× |
| prefill 4k         | 2.5× | 7.5× | 1.0× |
| prefill 16k        | 3.4× | 28×  | 1.7× |
| prefill 32k        | 4.1× | 51×  | 1.4× |
| **geomean**        | **2.2×** | **10×** | **0.9×** |

- **HCA** gains most: the fused single-pass kernel avoids Triton's fp32 scratch
  round-trip and stays ~flat (~30 µs) as the sequence grows, so the speedup
  widens to ~50× on long prefill.
- **CSA** leads throughout, scaling to ~4× at 32k.
- **indexer** (head=128, small per-token work) is launch-bound at small batch
  where Triton wins; HIP overtakes only at throughput scale (≥16k), ≈ parity
  overall on this mix.

## HCA notes (ratio=128)

HCA shares CSA's output half (RMSNorm + GPT-J RoPE + FP8 E4M3 + UE8M0 + packed
store) but the front half differs: 128 rows/boundary, no overlap, single state
region (state_width = head_dim), state block_size 8, boundary `(pos+1)%128==0`.

The compress path is **compact-plan + K-split**: `hca_build_plan` atomically
compacts boundary token indices into `plan[]` (order-independent — each entry
carries its token index and looks up its own `kv_slot_mapping[token_idx]`), then
`hca_compress_plan_ksplit<NW>` runs `grid = plan_capacity` blocks; the K=128 pool
is split across `NW` waves with an LDS cross-wave online-softmax merge, and wave 0
writes the output. `NW` auto-dispatches by estimated boundary count
(`<=640 -> 8`, else `4`). Single fused kernel (no fp32 scratch round-trip); the
trade-off is no head_dim split, so small-N (decode / short prefill) occupancy is
capped — competitive with aiter at large N, behind it at small N.

## Tests & benchmark

```bash
.venv/bin/python -m pytest tests/kernels/attention/test_dsv4_compress.py            # gfx950: runs; else skips
.venv/bin/python -m pytest tests/kernels/attention/test_dsv4_compress_arch_guard.py # gating (runs everywhere)
.venv/bin/python benchmarks/kernels/benchmark_dsv4_compress.py                      # perf (gfx950, run from repo root)
```

- `tests/kernels/attention/test_dsv4_compress.py` — byte-exact vs Triton FP32
  (CSA / HCA / indexer FP8) + indexer MXFP4 vs a torch front-end oracle.
- `tests/kernels/attention/test_dsv4_compress_arch_guard.py` — the gfx950 gating
  contract (platform-agnostic).
- `tests/kernels/attention/dsv4_compress_utils.py` — shared scenario /
  state-cache builder + runners + dequant/compare + MXFP4 oracle (not collected).
- `benchmarks/kernels/benchmark_dsv4_compress.py` — DVFS-robust, kernel-only,
  same-process HIP/Triton ratio.

> **Measurement note.** On a shared box the GPU clock idles to ~95 MHz and boosts
> to ~2.4 GHz; absolute µs swing wildly. Clocks are not locked — only
> **same-process ratios** are trustworthy.

> The table above is a headline summary of the final design. Rerun
> `benchmark_dsv4_compress.py` for fresh numbers on the target gfx950 system.
