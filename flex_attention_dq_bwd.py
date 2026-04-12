#@title Imports
import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np
import time
import statistics as stats
from util import make_jax_score_fn
import masks
import scores
from mha_reference import mha_bwd_reference
from jax_exp import _flash_attention_bwd_dq

dimension_numbers = (((1,), (1,)), ((), ()))
MIN_BLOCK_SIZE = 128
mask_value = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

def mha_reference(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
):
  logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
  if ab is not None:
    logits += ab
  if sm_scale != 1.0:
    logits *= sm_scale

  # --- causal masking (disabled for now but can enable later)
  mask = None
  
  logits = logits if mask is None else logits + jnp.where(mask, 0.0, -1e9)

  m = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - m[..., None])
  l = unnormalized.sum(axis=-1)
  weights = unnormalized / l[..., None]
  out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
  if save_residuals:
    return out, l, m
  return out

# def mha_bwd_reference(
#     q,
#     k,
#     v,
#     o,
#     do,
#     l,
#     m,
#     ab: jax.Array | None = None,
#     *,
#     causal: bool = False,
#     mask_value: float = None,
#     sm_scale: float = 1.0,
#     save_residuals: bool = True,
# ):
#   if mask_value is None:
#     mask_value = -1e9
#   logits = jnp.einsum(
#       "bhqc,bhkc->bhqk",
#       q.astype(jnp.float32),
#       k.astype(jnp.float32),
#   )

#   if sm_scale != 1.0:
#     logits *= sm_scale
#   mask = None

#   if causal:
#     print("causal")
#     _, _, q_seq_len, _ = q.shape
#     _, _, kv_seq_len, _ = k.shape
#     mask_shape = (q_seq_len, kv_seq_len)
#     row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
#     col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
#     causal_mask = (col_ids <= row_ids)[None, None, :, :]
#     mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
#   print(mask)
#   logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

#   unnormalized = jnp.exp(logits - m[..., None])
#   p = unnormalized / l[..., None]
#   dv = jnp.einsum("bhpt,bhpd->bhtd", p, do.astype(jnp.float32)).astype(v.dtype)

#   dp = jnp.einsum(
#       "bhpd,bhtd->bhpt", do.astype(jnp.float32), v.astype(jnp.float32)
#   )

#   di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)[
#       ..., None
#   ]  # [batch_size, num_heads, q_seq_len]

#   ds = (dp - di) * p
#   ds = ds * sm_scale

#   dk = jnp.einsum("bhsd,bhst->bhtd", q.astype(jnp.float32), ds).astype(k.dtype)
#   dq = jnp.einsum("bhst,bhtd->bhsd", ds, k.astype(jnp.float32)).astype(q.dtype)

#   # dab is just ds
#   dab = ds if ab is not None else None

#   return dq, dk, dv

def flex_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    o,
    do,
    d,
    *,
    block_b,
    block_q,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    debug: bool,
    score_fn = None,
    mask_fn=None, 
    block_mask_fn=None,
):
    # broadcast l,m,d
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    d = jnp.broadcast_to(d[..., None], (*d.shape, MIN_BLOCK_SIZE))

    # Preprocess contraction for bwd pass
    batch_size, head_num, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape

    # Grid specification
    # Removed block_q_major, utilizing block_q directly
    grid = (
        pl.cdiv(batch_size, block_b),
        head_num,
        pl.cdiv(q_seq_len, block_q),
        pl.cdiv(kv_seq_len, block_k_major),
    )

    # input qo
    def qo_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    # Replaced block_q_major with block_q
    qo_spec = pl.BlockSpec((block_b, 1, block_q, head_dim), qo_index_map)
    do_spec = qo_spec
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)
    assert o.ndim == len(qo_spec.block_shape)
    assert do.ndim == len(qo_spec.block_shape)

    # input kv
    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if block_mask_fn is not None:
            # Note: q_seq_index now refers to the block_q index, not a major block index
            next_kv_index = jax.lax.select(
                block_mask_fn(q_seq_index, kv_seq_index),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    kv_spec = pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    # input lm
    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    # Replaced block_q_major with block_q
    lm_spec = pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        return (batch_index, head_index, q_seq_index, kv_seq_index)

    dab_spec = (
      pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map)
      if ab is not None
      else None
    )

    def d_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)
    
    # Replaced block_q_major with block_q
    d_spec = pl.BlockSpec((batch_size, 1, block_q, MIN_BLOCK_SIZE), d_index_map)
    assert d_spec.block_shape is not None
    assert d.ndim == len(d_spec.block_shape)

    # Allocate scratch buffers
    # Replaced block_q_major with block_q
    dq_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
    scratch_shapes = [dq_scratch]

    dq_spec = pl.BlockSpec((block_b, 1, block_q, head_dim), qo_index_map)

    # in_spec specify
    in_specs = [
      qo_spec,      # q
      kv_spec,      # k
      kv_spec,      # v
      dab_spec,     # bias
      lm_spec,      # l
      lm_spec,      # m
      do_spec,      # do
      d_spec,       # d
    ]

    # out_spec specify
    out_specs = [
      dq_spec,
    ]

    out_shapes = [
      jax.ShapeDtypeStruct(q.shape, q.dtype),
    ]
    
    kernel = functools.partial(
        _flex_attention_bwd_dq,
        sm_scale = sm_scale,
        block_q = block_q,
        block_k = block_k,
        q_seq_len = q_seq_len,
        kv_seq_len = kv_seq_len,
        score_fn= score_fn,
        mask_fn = mask_fn,
        block_mask_fn = block_mask_fn
    )

    dq, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shapes,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
      ),
    )(q, k, v, ab, l, m, do, d)

    return dq

def _flex_attention_bwd_dq(q_tile_ref, *args, **kwargs):
    block_b = q_tile_ref.shape[0]
    # Create the real kernel from the factory
    kernel = flex_attention_bwd_dq_kernel

    for batch_idx in range(block_b):
        kernel(
            (batch_idx, 0),
            q_tile_ref,
            *args,
            **kwargs,
        )

def flex_attention_bwd_dq_kernel(
      batch_idx,
      q_tile_ref,
      k_tile_ref,
      v_tile_ref,
      ab_tile_ref,
      l_tile_ref,
      m_tile_ref,
      dO_tile_ref,
      di_tile_ref,
      dq_tile_ref,
      dq_scratch_ref,
      *,
      sm_scale,
      block_q,
      block_k,
      q_seq_len,
      kv_seq_len,
      score_fn,
      mask_fn,
      block_mask_fn
  ):

    kv_tile_idx = pl.program_id(axis = 3)
    # q_tile_idx is now directly the block_q index
    q_tile_idx = pl.program_id(axis = 2) 
    
    _, _, block_k_major, _ = k_tile_ref.shape

    block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)

    # Initialize scratch to zero at the start of the sequence scan
    @pl.when(kv_tile_idx == 0)
    def start_new_seq():
        dq_scratch_ref[batch_idx] = jnp.zeros(dq_scratch_ref.shape[2:], dq_scratch_ref.dtype)

    if block_mask_fn is None:
      should_run = True
    else:
      should_run = block_mask_fn(q_tile_idx, kv_tile_idx)
    
    # fix q and stream over kv blocks because we are doing accumulation over each q block
    @pl.when(should_run)
    def body():
        @pl.loop(0, block_k_major // block_k, unroll=True)
        def _body(j):
            start_kv = j * block_k

            q = q_tile_ref[batch_idx]
            k = k_tile_ref[(*batch_idx, pl.dslice(start_kv, block_k), slice(None))]
            v = v_tile_ref[(*batch_idx, pl.dslice(start_kv, block_k), slice(None))]

            l = l_tile_ref[batch_idx]
            m = m_tile_ref[batch_idx]
            dO = dO_tile_ref[batch_idx]
            D = di_tile_ref[batch_idx]

            dq_past = dq_scratch_ref[batch_idx]

            # S = QK^T
            if score_fn is not None:
                S, score_grad_fn = jax.vjp(score_fn, q, k)
            else:
                S = jax.lax.dot_general(q, k, dimension_numbers, preferred_element_type=jnp.float32)
                score_grad_fn = None
            S = S * sm_scale

            if mask_fn is not None:
                # Correct calculation for q_pos using block_q directly
                q_start = q_tile_idx * block_q
                k_start_major = kv_tile_idx * block_k_major
                k_start = k_start_major + start_kv

                # Ensure we broadcast for (Q, 1) and (1, K) to create (Q, K) mask
                q_pos = (q_start + jnp.arange(block_q, dtype=jnp.int32))
                k_pos = (k_start + jnp.arange(block_k, dtype=jnp.int32))

                token_keep_mask = mask_fn(q_pos, k_pos)  # [Bq, Bk] bool
                S = S + jnp.where(token_keep_mask, 0.0, -0.7 * float(jnp.finfo(jnp.dtype("float32")).max))

            # P(i,j) = exp(S(i,j)-L(i))
            unnormalized = jnp.exp(S - pltpu.repeat(m, block_k_repeats, axis=1))
            P = unnormalized / pltpu.repeat(l, block_k_repeats, axis=1)

            
            # dP(i,j) = dO(i)V(i)^T
            dP = jax.lax.dot_general(dO,v,dimension_numbers,preferred_element_type=jnp.float32)
            
            # dS(i,j) = P(i,j) * (dP(i,j)-D(i))
            dS = P * (dP  - pltpu.repeat(D, block_k_repeats, axis=1))
            dS = dS * sm_scale

            # dQ calculation
            if score_grad_fn is not None:
              dq_update, dk_update = score_grad_fn(dS)
            else:
              dq_update = jax.lax.dot_general(dS,k,(((1,), (0,)), ((), ())),preferred_element_type=jnp.float32)

            dq = dq_past + dq_update

            dq_scratch_ref[batch_idx] = dq.astype(dq_scratch_ref.dtype)

    # Store result when we reach the end of the KV sequence
    @pl.when(kv_tile_idx == (kv_seq_len // block_k_major) - 1)
    def store_res():
        dq_tile_ref[batch_idx] = dq_scratch_ref[batch_idx].astype(dq_tile_ref.dtype)

def flop_count_attention(b, h, q, k, d):
    return 4.0 * b * h * q * k * d

def benchmark(fn, args, iters=30, warmup=5, name="fn"):
    for _ in range(warmup):
        y = fn(*args)
        if isinstance(y, (tuple, list)):
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, y
            )
        else:
            y.block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn(*args)
        if isinstance(y, (tuple, list)):
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, y
            )
        else:
            y.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_t = sum(times) / len(times)
    med_t = stats.median(times)
    p10, p90 = np.percentile(np.array(times), [10, 90])

    print(f"[{name}] mean={mean_t*1e3:.2f} ms  median={med_t*1e3:.2f} ms  "
          f"p10={p10*1e3:.2f} ms  p90={p90*1e3:.2f} ms")

    return mean_t, med_t

def build_fns_for_bench(
    q, k, v,
    l, m, o, do, d,
    *,
    ab=None,
    sm_scale=1.0,
    causal=False,
    block_b=1,
    block_q=128,
    block_k_major=128,
    block_k=128,
    debug=False,
    score_fn=None,
    window_size=None,
    segment_ids=None,
    s2_stride=None,
    alibi_slope=None,
    mask_fn=None,
    block_mask_fn=None,
):
    batch_size, head_num, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape

    # ----------------- Reference backward -----------------
    ref_fn = functools.partial(
        mha_bwd_reference,
        ab=ab,
        sm_scale=sm_scale,
        score_fn=score_fn,
        causal=causal,
        window_size=window_size,
        segment_ids=segment_ids,
        s2_stride=s2_stride,
        alibi_slope=alibi_slope,
    )
    # Call as: ref_jit(q, k, v, o, do, l, m)
    ref_jit = jax.jit(ref_fn)

    # ----------------- FlexAttention backward (Pallas) -----------------
    flex_partial = functools.partial(
        flex_attention_bwd_dq,
        ab=ab,
        sm_scale=sm_scale,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        segment_ids=segment_ids,  # if you want to support it later
        debug=debug,
        score_fn=score_fn,
        mask_fn=mask_fn,
        block_mask_fn=block_mask_fn,
    )

    def flex_bwd_fn(q, k, v, l, m, o, do, d):
        out = flex_partial(q=q, k=k, v=v, l=l, m=m, o=o, do=do, d=d)
        # Normalize to just dq
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    flex_jit = jax.jit(flex_bwd_fn)

    # ----------------- Flash backward (reference Pallas kernel) -----------------
    flash_partial = functools.partial(
        _flash_attention_bwd_dq,
        ab=ab,
        segment_ids=segment_ids,
        block_q_major=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=-1e9,  # or your DEFAULT_MASK_VALUE
        debug=debug,
    )

    def flash_bwd_fn(q, k, v, l, m, o, do, d):
        # dq, ds = _flash_attention_bwd_dq(q, k, v, ab, segment_ids, l, m, do, di, ...)
        dq, _ = flash_partial(q, k, v, l=l, m=m, do=do, di=d)
        return dq

    flash_jit = jax.jit(flash_bwd_fn)

    return ref_jit, flex_jit, flash_jit


def run_bench_suite(
    q, k, v,
    l, m, o, do, d,
    *,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    causal=False,
    score_fn=None,
    window_size=None,
    segment_ids=None,
    s2_stride=None,
    alibi_slope=None,
    mask_fn=None,
    block_mask_fn=None,
):
    b, h, q_len, h_d = q.shape
    _, _, k_len, _ = k.shape

    gflops = flop_count_attention(b, h, q_len, k_len, h_d) / 1e9

    ref_jit, flex_jit, flash_jit = build_fns_for_bench(
        q, k, v,
        l, m, o, do, d,
        ab=None,
        sm_scale=sm_scale,
        causal=causal,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        debug=False,
        score_fn=score_fn,
        window_size=window_size,
        segment_ids=segment_ids,
        s2_stride=s2_stride,
        alibi_slope=alibi_slope,
        mask_fn=mask_fn,
        block_mask_fn=block_mask_fn,
    )

    print(f"\n== Benchmark config: b={b}, h={h}, q={q_len}, k={k_len}, d={h_d}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs")

    # ----------------- Reference timing -----------------
    t_mean_ref, t_med_ref = benchmark(
        ref_jit,
        (q, k, v, o, do, l, m),
        name="mha_reference_bwd[jit]",
    )
    print(f"  → Reference throughput: {gflops / t_med_ref:.2f} GFLOP/s")

    # ----------------- FlexAttention timing -----------------
    t_mean_flex, t_med_flex = benchmark(
        flex_jit,
        (q, k, v, l, m, o, do, d),
        name="flex_bwd_dq[jit]",
    )
    print(f"  → Flex throughput:      {gflops / t_med_flex:.2f} GFLOP/s")

    # ----------------- FlashAttention timing -----------------
    t_mean_flash, t_med_flash = benchmark(
        flash_jit,
        (q, k, v, l, m, o, do, d),
        name="flash_bwd_dq[jit]",
    )
    print(f"  → Flash throughput:     {gflops / t_med_flash:.2f} GFLOP/s")

    # ----------------- Numeric correctness -----------------
    dq_ref, dk_ref, dv_ref = ref_jit(q, k, v, o, do, l, m)
    dq_ref = dq_ref.block_until_ready()

    dq_flex = flex_jit(q, k, v, l, m, o, do, d)
    dq_flex = dq_flex.block_until_ready()
    print(f"diff dq_flex: {jnp.linalg.norm(dq_flex - dq_ref) / (jnp.linalg.norm(dq_ref) + 1e-6)}")
    dq_flash = flash_jit(q, k, v, l, m, o, do, d)
    dq_flash = dq_flash.block_until_ready()


    rel_err_flex = jnp.linalg.norm(dq_flex - dq_ref) / (jnp.linalg.norm(dq_ref) + 1e-6)
    rel_err_flash = jnp.linalg.norm(dq_flash - dq_ref) / (jnp.linalg.norm(dq_ref) + 1e-6)

    print(f"Numeric diff Flex  vs Ref (Relative L2): {rel_err_flex}")
    print(f"Numeric diff Flash vs Ref (Relative L2): {rel_err_flash}")

    return {
        "ref_ms_med": t_med_ref * 1e3,
        "flex_ms_med": t_med_flex * 1e3,
        "flash_ms_med": t_med_flash * 1e3,
        "ref_gflops": gflops / t_med_ref,
        "flex_gflops": gflops / t_med_flex,
        "flash_gflops": gflops / t_med_flash,
        "rel_l2_flex": float(rel_err_flex),
        "rel_l2_flash": float(rel_err_flash),
    }



import jax
import jax.numpy as jnp
from jax import random
# import pandas as pd

# --- TPU v5e Specs (Approximate) ---
TPU_PEAK_TFLOPS = 197.0
TPU_PEAK_BW = 819.0  # GB/s

def get_theoretical_metrics_fwd(b, h, l, d, causal=True, dtype_bytes=2):
    """Same as your forward model: ~4 * B * H * L^2 * D FLOPs."""
    total_elements = 4 * (b * h * l * d)          # Q, K, V, O
    total_bytes = total_elements * dtype_bytes

    total_flops = 4 * b * h * (l * l) * d
    if causal:
        total_flops /= 2.0
    return total_flops, total_bytes


def get_theoretical_metrics_bwd(b, h, l, d, causal=True, dtype_bytes=2):
    """
    Approximate FLOP/byte model for the backward pass.

    FLOPs:
      - Backward of attention is more expensive than forward.
      - A simple and reasonable approximation is 2x forward FLOPs.
    IO:
      - Read: Q, K, V, O, dO, l, m     (~7 tensors)
      - Write: dQ, dK, dV              (~3 tensors)
      For roofline we just count the main tensors and use 7 * B * H * L * D.
    """
    fwd_flops, _ = get_theoretical_metrics_fwd(b, h, l, d, causal=causal,
                                               dtype_bytes=dtype_bytes)
    total_flops = 2.0 * fwd_flops  # heuristic: bwd ≈ 2× fwd

    total_elements = 7 * (b * h * l * d)  # Q,K,V,O,dO,l,m (approx)
    total_bytes = total_elements * dtype_bytes

    return total_flops, total_bytes


def roofline():
    key = random.PRNGKey(0)

    # Constants
    BATCH = 1
    HEADS = 8
    DIM = 128

    # Block sizes (keep constant)
    BLOCK_Q = 1024
    BLOCK_K_MAJOR = 1024
    BLOCK_K = 1024

    # Sequence length sweep (same style as fwd)
    SEQ_LENS = [1024 * (2 ** i) for i in range(4)]  # 1k, 2k, 4k, 8k, 16k

    results_data = []

    print(f"{'SeqLen':<10} | {'Time(ms)':<10} | {'TFLOP/s':<10} | {'Intensity':<10}")
    print("-" * 55)

    def my_score(q, k):
        # Tile-local score; wrapped by make_jax_score_fn
        return jnp.einsum("qd,kd->qk", q, k)

    for L in SEQ_LENS:
        # 1. Generate inputs
        k1, k2, k3, k4, key = random.split(key, 5)

        # Use bf16 for inputs to match TPU execution mode
        q = random.normal(k1, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)
        k = random.normal(k2, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)
        v = random.normal(k3, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)

        # 2. Forward pass (reference) to get O, l, m
        sm_scale = 1.0
        o, l, m = mha_reference(
            q.astype(jnp.float32),
            k.astype(jnp.float32),
            v.astype(jnp.float32),
            sm_scale=sm_scale,
            save_residuals=True,
        )

        # 3. Generate dO and compute the scalar "d" (di) term
        do = random.normal(k4, o.shape, dtype=jnp.bfloat16)
        d = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)

        # 4. Build mask + score functions
        # mask_fn, block_mask_fn = make_causal_mask_fns(
        #     block_q=BLOCK_Q, block_k_major=BLOCK_K_MAJOR
        # )
        jax_score = make_jax_score_fn(my_score)

        # 5. Jitted backward kernel
        def bwd_call(q_, k_, v_, o_, l_, m_, do_, d_):
            return flex_attention_bwd_dq(
                q=q_, k=k_, v=v_,
                ab=None,
                segment_ids=None,
                l=l_, m=m_, o=o_,
                do=do_, d=d_,
                block_b=1,
                block_q=BLOCK_Q,
                block_k_major=BLOCK_K_MAJOR,
                block_k=BLOCK_K,
                sm_scale=sm_scale,
                debug=False,
                score_fn=None,
                mask_fn=None,
                block_mask_fn=None,
            )

        bwd_jit = jax.jit(bwd_call)

        # 6. Benchmark backward kernel only
        _, time_sec = benchmark(
            bwd_jit,
            (q, k, v, o, l, m, do, d),
            name=f"flex_bwd_dq_L{L}",
        )

        # 7. Theoretical FLOPs and bytes for backward
        flops, bytes_moved = get_theoretical_metrics_bwd(
            BATCH, HEADS, L, DIM, causal=True, dtype_bytes=2
        )

        tflops_per_sec = (flops / 1e12) / time_sec
        intensity = flops / bytes_moved

        print(f"{L:<10} | {time_sec * 1e3:<10.2f} | {tflops_per_sec:<10.2f} | {intensity:<10.2f}")

        results_data.append({
            "SeqLen": L,
            "Time_Sec": float(time_sec),
            "TFLOPs": float(tflops_per_sec),
            "Intensity": float(intensity),
        })

    # 8. Save CSV for backward roofline plot
    df = pd.DataFrame(results_data)
    df.to_csv("roofline_data_bwd.csv", index=False)
    print("\nSaved backward sweep data to roofline_data_bwd.csv")


# ==============================================================================
# Helper to Generate Fake Document IDs (same as fwd)
# ==============================================================================
def generate_doc_lengths(total_len, num_docs, seed=0):
    """
    Generates a list of random document lengths that sum exactly to total_len.
    Ensures no document has 0 length.
    """
    np.random.seed(seed)
    
    if num_docs <= 0:
        return [total_len]
    if num_docs > total_len:
        raise ValueError("Cannot have more documents than tokens!")

    # 1. Generate split points
    splits = np.sort(
        np.random.choice(range(1, total_len), num_docs - 1, replace=False)
    )
    
    # 2. Add start (0) and end (total_len)
    boundaries = np.concatenate(([0], splits, [total_len]))
    
    # 3. Calculate lengths (distance between cuts)
    lengths = np.diff(boundaries)
    
    return lengths.tolist()


# ==============================================================================
# Main Execution Loop for Backward Benchmark
# ==============================================================================
def main():
    print("=== FlexAttention Backward Masking Benchmark & Verification ===\n")
    
    # 1. Hardware / Data Config
    # -------------------------
    key = random.PRNGKey(0)
    batch, heads = 1, 8
    q_len, kv_len = 4096, 4096
    head_dim = 128
    
    # Block sizes (must match your bwd kernel tiling)
    block_q = 1024
    block_k_major = 1024
    block_k = 1024

    # 2. Generate Inputs (BF16 for TPU speed)
    # -------------------------
    print(f"Generating inputs: B={batch}, H={heads}, L={q_len}, D={head_dim} (BF16)...")
    # Keep one extra key for later use
    k1, k2, k3, key = random.split(key, 4)
    q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.bfloat16)
    k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.bfloat16)
    v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.bfloat16)

    # 3. Define Test Cases (same semantics as fwd)
    # -------------------------
    test_cases = []

    # --- Case A: Standard Causal ---
    test_cases.append({
        "name": "Causal Attention",
        "factory": masks.make_causal_mask_fns,
        "factory_args": (),   # no extra args
        "ref_args": {"causal": True},
    })

    # --- Case B: Sliding Window ---
    window_size = 1024
    test_cases.append({
        "name": f"Sliding Window (W={window_size})",
        "factory": masks.make_sliding_window_mask_fns,
        "factory_args": (window_size,),
        "ref_args": {"causal": False, "window_size": window_size},
    })
    
    # # --- Case C: Jagged Documents (Randomized) ---
    # doc_lengths = generate_doc_lengths(total_len=q_len, num_docs=5, seed=42)
    # print(f"Generated Doc Lengths: {doc_lengths}")

    # # Build segment IDs for the reference (shape [B, L])
    # ref_ids_list = []
    # for i, length in enumerate(doc_lengths):
    #     ref_ids_list.append(jnp.full((length,), i, dtype=jnp.int32))
    # jagged_ids_ref = jnp.concatenate(ref_ids_list)
    # jagged_ids_ref = jnp.tile(jagged_ids_ref[None, :], (batch, 1))

    # test_cases.append({
    #     "name": f"Jagged Masking ({len(doc_lengths)} Docs)",
    #     "factory": masks.make_jagged_mask_fns,
    #     "factory_args": (doc_lengths,),
    #     "ref_args": {
    #         "causal": True,
    #         "segment_ids": jagged_ids_ref,
    #     },
    # })

    # --- Case D: ALiBi (Score Function) ---
    alibi_fn = make_jax_score_fn(
        scores.make_alibi_score_fn(slope=0.5)
    )
    test_cases.append({
        "name": "ALiBi Attention",
        "factory": lambda *args: (None, None),  # no masks: score_fn only
        "factory_args": (),
        "ref_args": {
            "score_fn": alibi_fn,
            "alibi_slope": 0.5,
            "causal": False,
        },
    })

    # --- Case E: Tanh Soft-Capping (Score Function) ---
    tanh_fn = make_jax_score_fn(
        scores.make_softcap_score_fn(cap=30.0)
    )
    test_cases.append({
        "name": "Tanh Soft-Capping",
        "factory": lambda *args: (None, None),
        "factory_args": (),
        "ref_args": {
            "score_fn": tanh_fn,
            "causal": False,
        },
    })

    # 4. Run Loop over Mask / Score Configurations
    # -------------------------
    for case in test_cases:
        print("\n" + "=" * 60)
        print(f"RUNNING (Backward): {case['name']}")
        print("=" * 60)

        # A. Build the masks
        factory_fn = case["factory"]
        extra_args = case["factory_args"]

        try:
            # NOTE: our factories are defined as:
            #   causal:  make_causal_mask_fns(block_q, block_k_major)
            #   others:  make_*_mask_fns(block_q, block_k, ...)
            # Passing block_k_major as "block_k" for the non-causal cases is OK
            # as long as it matches the tiling used in the kernel.
            mask_fn, block_mask_fn = factory_fn(
                block_q, block_k_major, *extra_args
            )
        except AttributeError:
            print(f"Skipping {case['name']} (factory not found in masks.py)")
            continue

        # B. Prepare arguments for backward benchmark
        current_args = {
            "sm_scale": 1.0,
            "block_b": 1,
            "block_q": block_q,
            "block_k_major": block_k_major,
            "block_k": block_k,
            "mask_fn": mask_fn,
            "block_mask_fn": block_mask_fn
        }
        current_args.update(case["ref_args"])

        sm_scale = 1.0
        o, l, m = mha_reference(
            q.astype(jnp.float32),
            k.astype(jnp.float32),
            v.astype(jnp.float32),
            sm_scale=sm_scale,
            save_residuals=True,
        )

        # 3. Generate dO and compute the scalar "d" (di) term
        #    IMPORTANT: use a PRNG key, NOT the tensor k
        key, key_do = random.split(key)
        do = random.normal(key_do, o.shape, dtype=jnp.bfloat16)
        d = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)

        # C. Run backward benchmark suite
        results = run_bench_suite(
            q, k, v, l, m, o, do, d,
            **current_args,
        )

        print("Backward summary:", results)

    print("\n=== All Backward Tests Completed ===")


if __name__ == "__main__":
    main()
