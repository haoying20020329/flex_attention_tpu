#@title Imports
import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.extend import core
from jax._src.util import safe_map
import statistics as stats
import time
import numpy as np
import masks
from jax import random

from constants import dimension_numbers, MIN_BLOCK_SIZE
from mha_reference import mha_reference, mha_bwd_reference
from jax_exp import mha_reference_no_custom_vjp,_flash_attention_impl


def _inline_jaxpr_score(q, k, closed_jaxpr):
  jaxpr = closed_jaxpr.jaxpr
  literals = closed_jaxpr.literals
  env = {}

  def read(var):
      if isinstance(var, core.Literal):
          return var.val
      return env[var]

  def write(var, val):
      if not hasattr(val, 'dtype'):
          val = jnp.asarray(val)
      env[var] = val

  write(jaxpr.invars[0], q)
  write(jaxpr.invars[1], k)
  for var, val in zip(jaxpr.constvars, literals):
      write(var, val)

  for eqn in jaxpr.eqns:
      invals = [read(v) for v in eqn.invars]
      outvals = eqn.primitive.bind(*invals, **eqn.params)
      if not eqn.outvars:
          continue
      if eqn.primitive.multiple_results:
          for var, val in zip(eqn.outvars, outvals):
              write(var, val)
      else:
          write(eqn.outvars[0], outvals)

  return read(jaxpr.outvars[0])

def _flash_attention_kernel(q_tile_ref, *args, score_jaxpr=None,mask_fn=None,block_mask_fn=None, **kwargs):
    """Connects _flash_attention_impl to the generated kernel."""
    block_b = q_tile_ref.shape[0]

    kernel = make_flash_attention_kernel(
        score_jaxpr=score_jaxpr,
        mask_fn=mask_fn,
        block_mask_fn=block_mask_fn
    )

    for batch_idx in range(block_b):
        kernel(
            (batch_idx, 0),
            q_tile_ref,
            *args,
            **kwargs,
        )

def _flex_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
    block_mask_fn,
    mask_fn,
    score_fn,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape

  grid = (
      pl.cdiv(batch_size, block_b),
      num_heads,
      pl.cdiv(q_seq_len, block_q),
      kv_seq_len // block_k_major,
  )

  def q_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if block_mask_fn is not None:
      next_kv_index = jax.lax.select(
          block_mask_fn(q_seq_index, kv_seq_index),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    next_q_index = q_seq_index
    next_kv_index = kv_seq_index
    return (batch_index, head_index, next_q_index, next_kv_index)

  def o_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)



  kernel = functools.partial(
      _flash_attention_kernel,
      causal=causal,
      sm_scale=sm_scale,
      block_k=block_k,
      kv_seq_len=kv_seq_len,
      score_jaxpr=score_fn,
      block_mask_fn=block_mask_fn,
      mask_fn=mask_fn
  )


  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

  if block_k != kv_seq_len:
    m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    scratch_shapes = []

  if save_residuals:
    out_specs = [
        *out_specs,
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
    ]
    l = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = (*out_shape, l, m)
  else:
    out_specs = [*out_specs, None, None]
    out_shape = (*out_shape, None, None)

  ab_block_spec = (
      pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map)
      if ab is not None else None)

  in_specs = [
      pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      ab_block_spec,
  ]

  o, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shape,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
      ),
  )(q, k, v, ab)

  if save_residuals:
    l, m = (v[..., 0] for v in aux[-2:])
    return (o, l, m)
  else:
    return o
  

def make_flash_attention_kernel(mask_fn=None, block_mask_fn=None, score_jaxpr=None):
  """Factory returning a kernel with an optional custom mask function."""
  def flash_attention_fwd_kernel(
      batch_idx,
      q_tile_ref,
      k_tile_ref,
      v_tile_ref,
      ab_tile_ref,
      O_tile_ref,
      l_tile_ref,
      m_tile_ref,
      m_scratch_ref,
      l_scratch_ref,
      O_scratch_ref,
      *,
      causal,
      sm_scale,
      block_k,
      kv_seq_len,
  ):
    block_k_major = k_tile_ref.shape[2]
    head_dim = k_tile_ref.shape[3]
    block_q = q_tile_ref.shape[2]
    kv_seq_idx = pl.program_id(3)
    q_seq_idx = pl.program_id(2)   

    @pl.when(kv_seq_idx == 0)
    def start_new_seq():
      m_scratch_ref[batch_idx] = jnp.full(
          m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
      l_scratch_ref[batch_idx] = jnp.zeros(
          l_scratch_ref.shape[2:], jnp.float32)
      O_scratch_ref[batch_idx] = jnp.zeros(
          O_scratch_ref.shape[2:], jnp.float32)

    if block_mask_fn is None:
      should_run = True
    else:
      should_run = block_mask_fn(q_seq_idx, kv_seq_idx)

    @pl.when(should_run)
    def body():
      @pl.loop(0, block_k_major, step=block_k, unroll=True)
      def _body(start_k):
        m_past = m_scratch_ref[batch_idx]
        l_past = l_scratch_ref[batch_idx]
        O_past = O_scratch_ref[batch_idx]
        k_ref = k_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
        q_ref = q_tile_ref[batch_idx]
        block_q = q_ref.shape[0]
        

        if score_jaxpr is None:
            S = jax.lax.dot_general(
                q_ref, k_ref, dimension_numbers,
                preferred_element_type=jnp.float32
            )
            S = S * sm_scale
        else:
            S = score_jaxpr(q_ref, k_ref)

        # S = jax.lax.dot_general(
        #         q_ref, k_ref, dimension_numbers,
        #         preferred_element_type=jnp.float32
        #     )

        if ab_tile_ref is not None:
          ab = ab_tile_ref[
              (*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))
          ].astype(jnp.float32)
          S += ab

        # if causal:
        #     q_start = q_seq_idx * q_ref.shape[0]  # block_q
        #     k_start = kv_seq_idx * block_k_major + start_k

        #     q_pos = (q_start + jnp.arange(q_ref.shape[0]))[:, None]
        #     k_pos = (k_start + jnp.arange(block_k))[None, :]

        #     causal_mask = k_pos > q_pos
        #     S = jnp.where(causal_mask, -1e9, S)

        if mask_fn is not None:
          q_start = q_seq_idx * block_q
          k_start_major = kv_seq_idx * block_k_major
          k_start = k_start_major + start_k

          q_pos = (q_start + jnp.arange(block_q, dtype=jnp.int32))  # [Bq]
          k_pos = (k_start + jnp.arange(block_k, dtype=jnp.int32))  # [Bk]

          token_keep_mask = mask_fn(q_pos, k_pos)  # [Bq, Bk] bool
          # S = jnp.where(token_keep_mask, S, -1e9)
          S = S + jnp.where(token_keep_mask, 0.0, -0.7 * float(jnp.finfo(jnp.dtype("float32")).max))

        m_cur = jnp.max(S, axis=1)[:, None]
        m_next = jnp.maximum(m_cur, m_past)
        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(
              f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
          )

        P = jnp.exp(S - pltpu.repeat(m_next, block_k_repeats, 1))
        l_corr = jnp.exp(m_past - m_next) * l_past
        l_next = l_corr + jnp.sum(P, axis=1)[:, None]

        head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

        l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
        l_scratch_ref[batch_idx] = l_next
        m_scratch_ref[batch_idx] = m_next

        l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
        v_ref = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
        o_curr = jax.lax.dot(
            P.astype(v_ref.dtype), v_ref,
            preferred_element_type=jnp.float32
        )
        O_scratch_ref[batch_idx] = O_past * l_broadcast(l_corr) + o_curr
        O_scratch_ref[batch_idx] *= l_broadcast(l_next_inv_safe)

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_res():
      O_tile_ref[batch_idx] = O_scratch_ref[batch_idx].astype(O_tile_ref.dtype)
      if (m_tile_ref is not None) and (l_tile_ref is not None):
        l_tile_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_tile_ref.dtype)
        m_tile_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_tile_ref.dtype)
  return flash_attention_fwd_kernel


# ============================================================
# FLOP COUNT
# ============================================================

def flop_count_attention(b, h, q, k, d):
    return 4.0 * b * h * q * k * d


# ============================================================
# GENERIC BENCH FUNCTION
# ============================================================

def benchmark(fn, args, iters=30, warmup=5, name="fn"):
    # ---- Warmup ----
    for _ in range(warmup):
        out = fn(*args)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            out
        )

    # ---- Timed runs ----
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            out
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    if not times:
        return 0.0, 0.0

    mean_t = sum(times) / len(times)
    med_t = stats.median(times)
    p10, p90 = np.percentile(times, [10, 90])

    print(f"[{name}] mean={mean_t*1e3:.2f} ms  median={med_t*1e3:.2f} ms  "
          f"p10={p10*1e3:.2f} ms  p90={p90*1e3:.2f} ms")

    return mean_t, med_t


# ============================================================
# DIFF TOOL
# ============================================================

def compute_diff(ref_out, test_out):
    if not isinstance(ref_out, (tuple, list)):
        ref_out = (ref_out,)
    if not isinstance(test_out, (tuple, list)):
        test_out = (test_out,)

    diffs = {}
    for i, (r, t) in enumerate(zip(ref_out, test_out)):
        if r.shape != t.shape:
            # Squeeze simplified dims if necessary or just warn
            if r.size == t.size:
                t = t.reshape(r.shape)
            else:
                raise ValueError(f"Shape mismatch: ref {r.shape} vs test {t.shape}")

        diffs[i] = float(
            jnp.linalg.norm(t - r) / (jnp.linalg.norm(r) + 1e-6)
        )

    return diffs


# ============================================================
# FN BUILDER
# ============================================================

def build_fns_for_bench(
    q, k, v,
    *,
    ab=None,
    sm_scale=1.0,
    save_residuals=True,
    causal=False,
    block_b=1,
    block_q=128,
    block_k_major=128,
    block_k=128,
    debug=False,
    score_fn=None,
    mask_fn=None,           
    block_mask_fn=None,     
    window_size: int | None = None,     # Sliding Window
    segment_ids: jax.Array | None = None, # Document Masking
    s2_stride: int | None = None,         # S2 Attention
    alibi_slope: float | None = None,
    which=("ref", "flash", "flash_ref"),
):
    out = {}

    # ========================================================
    # REFERENCE MHA
    # ========================================================
    if "ref" in which:
        # We pass ALL the configuration parameters to the reference
        # so it matches the logic inside mask_fn exactly.
        ref_fwd = functools.partial(
            mha_reference,
            ab=None,
            sm_scale=sm_scale,
            save_residuals=True,
            score_fn=score_fn,
            causal=causal,
            window_size=window_size,
            segment_ids=segment_ids,  # <--- Passed to Ref
            s2_stride=s2_stride,       # <--- Passed to Ref
            alibi_slope = alibi_slope
        )
        # segment_ids is an array, not static, so we don't put it in static_argnames
        out["ref_fwd_jit"] = jax.jit(ref_fwd, static_argnames=("score_fn",))

    # ========================================================
    # FLASH ATTENTION PALLAS KERNEL
    # ========================================================
    if "flash" in which:
        print(f"Compiling Flash Kernel with: Mask={mask_fn is not None}, BlockMask={block_mask_fn is not None}")
        
        # The Flash Kernel relies on the closures (mask_fn/block_mask_fn) 
        # to handle the logic for window/docs/s2. We don't pass s2_stride directly 
        # unless the kernel expects it, but we assume it's baked into mask_fn.
        flash_fwd = functools.partial(
            _flex_attention_impl,
            ab=ab,
            segment_ids=None, # Usually None for Pallas as it's handled in mask_fn
            save_residuals=True,
            causal=causal,
            sm_scale=sm_scale,
            block_b=block_b,
            block_q=block_q,
            block_k_major=block_k_major,
            block_k=block_k,
            debug=debug,
            score_fn=score_fn,
            mask_fn=mask_fn,             
            block_mask_fn=block_mask_fn
        )
        
        out["flash_fwd_jit"] = jax.jit(
            flash_fwd, 
            static_argnames=("score_fn", "mask_fn", "block_mask_fn")
        )

    # ========================================================
    # FLASH ATTENTION REFERENCE (Loop-based)
    # ========================================================
    if "flash_ref" in which:
        flash_ref_fwd = functools.partial(
            _flash_attention_impl,
            ab=ab,
            segment_ids=segment_ids,
            save_residuals=True,
            causal=causal,
            sm_scale=sm_scale,
            block_b=block_b,
            block_q=block_q,
            block_k_major=block_k_major,
            block_k=block_k,
            debug=debug,
        )
        out["flash_ref_fwd_jit"] = jax.jit(flash_ref_fwd)

    return out


# ============================================================
# MAIN BENCH SUITE
# ============================================================

def run_bench_suite(
    q, k, v,
    *,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    causal=False,
    score_fn=None,
    mask_fn=None,         
    block_mask_fn=None,   
    window_size: int | None = None,
    segment_ids: jax.Array | None = None,
    s2_stride: int | None = None,
    alibi_slope: float | None = None,
    which=("ref", "flash", "flash_ref"),
):
    b, h, q_len, d = q.shape
    _, _, k_len, _ = k.shape

    gflops = flop_count_attention(b, h, q_len, k_len, d) / 1e9
    
    compiled = build_fns_for_bench(
        q, k, v,
        sm_scale=sm_scale,
        causal=causal,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        score_fn=score_fn,
        mask_fn=mask_fn,             
        block_mask_fn=block_mask_fn, 
        window_size=window_size,     # <--- Passed
        segment_ids=segment_ids,     # <--- Passed
        s2_stride=s2_stride,         # <--- Passed
        alibi_slope=alibi_slope,
        which=which,
    )

    print(f"\n== Benchmark config: b={b}, h={h}, q={q_len}, k={k_len}, d={d}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs")
    if block_mask_fn:
        print("Note: FLOP count does not account for skipped blocks (sparsity).")

    results = {}

    # ======================================================
    # FORWARD BENCHMARKS
    # ======================================================
    for name in ("ref_fwd_jit", "flash_fwd_jit", "flash_ref_fwd_jit"):
        if name in compiled:
            fn = compiled[name]
            t_mean, t_med = benchmark(fn, (q, k, v), iters=10, name=name)
            if t_med > 0:
                print(f"  → FWD Throughput: {gflops/t_med:.2f} GFLOP/s\n")
            results[name] = (t_mean, t_med)
    
    print("--- Numeric Accuracy (vs ref_fwd_jit) ---")
    
    ref_target = "ref_fwd_jit"
    ref_fwd_out = None

    if ref_target in compiled:
        try:
            ref_fwd_out = compiled[ref_target](q, k, v)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), ref_fwd_out)
        except Exception as e:
            print(f"Reference run failed: {e}")

    if ref_fwd_out:
        for name, fn in compiled.items():
            if name == ref_target: continue 
            # # Skip flash_ref if it doesn't support the new masks, to avoid noise
            # if name == "flash_ref_fwd_jit" and (window_size or segment_ids is not None):
            #      print(f"[{name}] Skipped accuracy check (implementation may not support new masks)")
            #      continue

            try:
                test_out = fn(q, k, v)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), test_out)

                diff_res = compute_diff(ref_fwd_out, test_out)
                print(f"[{name}] vs [{ref_target}]: {diff_res}")
            except Exception as e:
                print(f"[{name}] Validation failed: {e}")
    else:
        print("Skipping accuracy check (no reference output).")

    return results

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
    # We pick (num_docs - 1) cuts in the sequence.
    # range(1, total_len) ensures we don't cut at 0 or end, avoiding 0-length docs.
    splits = np.sort(np.random.choice(range(1, total_len), num_docs - 1, replace=False))
    
    # 2. Add start (0) and end (total_len)
    boundaries = np.concatenate(([0], splits, [total_len]))
    
    # 3. Calculate lengths (distance between cuts)
    lengths = np.diff(boundaries)
    
    return lengths.tolist()

# ==============================================================================
# Main Execution Loop
# ==============================================================================
def main():
    print("=== FlexAttention Masking Benchmark & Verification ===\n")
    
    # 1. Hardware / Data Config
    # -------------------------
    key = random.PRNGKey(0)
    batch, heads = 1, 1
    q_len, kv_len = 16384, 16384
    head_dim = 128
    
    # Block Sizes (Must match what fits in your VMEM)
    block_q = 1024
    block_k_major = 1024
    block_k = 1024

    # 2. Generate Inputs (BF16 for TPU Speed)
    # -------------------------
    print(f"Generating inputs: B={batch}, H={heads}, L={q_len}, D={head_dim} (BF16)...")
    k1, k2, k3, k4 = random.split(key, 4)
    q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.bfloat16)
    k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.bfloat16)
    v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.bfloat16)

    # 3. Define Test Cases
    # -------------------------
    # Each case is a dictionary defining how to build the mask and what to tell the reference.
    
    test_cases = []

    # --- Case A: Standard Causal ---
    test_cases.append({
        "name": "Causal Attention",
        "factory": masks.place_holder,
        "factory_args": (), # No extra args needed for causal factory
        "ref_args": {"causal": False}
    })

    # # --- Case B: Sliding Window ---
    # window_size = 1024
    # test_cases.append({
    #     "name": f"Sliding Window (W={window_size})",
    #     "factory": masks.make_sliding_window_mask_fns,
    #     "factory_args": (window_size,), # Pass window_size to factory
    #     "ref_args": {"causal": False, "window_size": window_size}
    # })
    
    # # --- Case C: Jagged Documents (Randomized) ---
    # # Generate 5 random document lengths that sum to 8192
    # doc_lengths = generate_doc_lengths(total_len=q_len, num_docs=5, seed=42)
    # print(f"Generated Doc Lengths: {doc_lengths}") 
    # # Example Output: [1200, 350, 4000, 2000, 642]

    # # 1. Create Reference IDs for validation
    # #    (Builds the array [0,0,.., 1,1,.., 2,2,..])
    # ref_ids_list = []
    # for i, length in enumerate(doc_lengths):
    #     ref_ids_list.append(jnp.full((length,), i, dtype=jnp.int32))
    
    # jagged_ids_ref = jnp.concatenate(ref_ids_list)
    # jagged_ids_ref = jnp.tile(jagged_ids_ref[None, :], (batch, 1))

    # test_cases.append({
    #     "name": f"Jagged Masking ({len(doc_lengths)} Docs)",
    #     "factory": masks.make_jagged_mask_fns,
    #     # Pass the list of lengths to the factory
    #     "factory_args": (doc_lengths,),  
    #     # Pass the ID array to the reference
    #     "ref_args": {"causal": True, "segment_ids": jagged_ids_ref}
    # })

    # # --- Case D: ALiBi (Score Function) ---
    # # Returns (None, None) for masks, passes score_fn to ref_args
    
    # alibi_fn = util.make_jax_score_fn(scores.make_alibi_score_fn(slope=0.5))
    # test_cases.append({
    #     "name": "ALiBi Attention",
    #     "factory": lambda *args: (None, None), # Dummy factory
    #     "factory_args": (),
    #     "ref_args": { "score_fn": alibi_fn,"alibi_slope":0.5}
    # })

    # # --- Case E: Tanh Soft-Capping (Score Function) ---
    # tanh_fn = util.make_jax_score_fn(scores.make_softcap_score_fn(cap=30.0))
    # test_cases.append({
    #     "name": "Tanh Soft-Capping",
    #     "factory": lambda *args: (None, None),
    #     "factory_args": (),
    #     "ref_args": { "score_fn": tanh_fn}
    # })

    # 4. Run Loop
    # -------------------------
    for case in test_cases:
        print(f"\n" + "="*60)
        print(f"RUNNING: {case['name']}")
        print("="*60)

        # A. Build the Masks
        # Unpack factory args (block sizes + specific args)
        factory_fn = case['factory']
        extra_args = case['factory_args']
        
        try:
            mask_fn, block_mask_fn = factory_fn(block_q, block_k_major, *extra_args)
        except AttributeError:
            print(f"Skipping {case['name']} (Factory function not found in masks.py)")
            continue

        # B. Prepare Reference Arguments
        # Start with defaults
        current_ref_args = {
            "sm_scale": 1.0,
            "block_b": 1,
            "block_q": block_q,
            "block_k_major": block_k_major,
            "block_k": block_k,
            "which": ["flash", "ref","flash_ref"], # Compare Flash vs Reference
            "mask_fn": mask_fn,
            "block_mask_fn": block_mask_fn,
        }
        # Update with case-specific args (causal=True, window_size=..., etc.)
        current_ref_args.update(case["ref_args"])

        # C. Run Benchmark
        results = run_bench_suite(
            q, k, v,
            **current_ref_args
        )
        
        # D. Print Quick Status
        # Check diffs in the output (if printed by benchmark)
        # Note: benchmark.py prints the diffs automatically.

    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main()


# import os
# import shutil
# import jax
# import jax.numpy as jnp
# from jax import random

# # Assuming these are imported from your modules
# import benchmark 
# from main import make_causal_mask_fns 

# def main():
#     # --- 1. SETUP ---
#     key = random.PRNGKey(0)
#     # Increased sequence length to ensuring heavy compute
#     batch, heads, q_len, dim = 1, 1, 25600, 128
    
#     # Generate Inputs
#     # CRITICAL: Use bfloat16 for TPU. float32 is slow; float16 crashes.
#     k1, k2, k3 = random.split(key, 3)
#     q = random.normal(k1, (batch, heads, q_len, dim), dtype=jnp.bfloat16)
#     k = random.normal(k2, (batch, heads, q_len, dim), dtype=jnp.bfloat16)
#     v = random.normal(k3, (batch, heads, q_len, dim), dtype=jnp.bfloat16)

#     # Get Functions
#     mask_fn, block_mask_fn = make_causal_mask_fns(1024, 1024)
#     bench_fns = benchmark.build_fns_for_bench(
#         q, k, v,
#         sm_scale=1.0, block_b=1, block_q=1024, block_k_major=1024, block_k=512,
#         causal=True, mask_fn=mask_fn, block_mask_fn=block_mask_fn,
#         which=["flash"]
#     )
#     flash_jit = bench_fns["flash_fwd_jit"]

#     # --- 2. WARMUP ---
#     print("Warming up (compiling)...")
#     warmup_out = flash_jit(q, k, v)
#     jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_out)
#     print("Warmup done.")

#     # --- 3. PROFILE ---
#     trace_dir = "/tmp/tpu_profile"
    
#     if os.path.exists(trace_dir):
#         shutil.rmtree(trace_dir)

#     print(f"Starting trace... saving to {trace_dir}")
#     jax.profiler.start_trace(trace_dir)

#     # Run loop
#     for _ in range(100):
#         # FIX IS HERE: Call the function fresh every time!
#         out = flash_jit(q, k, v)
        
#         # Block on the NEW output, not the warmup output
#         jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)

#     jax.profiler.stop_trace()
#     print("Trace finished!")
    
#     # Optional: Snapshot memory usage to verify you aren't leaking HBM
#     jax.profiler.save_device_memory_profile(f"{trace_dir}/memory.prof")

# if __name__ == "__main__":
#     main()

# import jax
# import jax.numpy as jnp
# from jax import random
# import pandas as pd
# import time

# # Import your existing modules
# import benchmark 
# from main import make_causal_mask_fns

# # --- TPU v5e Specs (Approximate) ---
# # Peak Compute (BF16 Matrix Mul): ~197 TFLOPS
# # Peak Memory Bandwidth: ~819 GB/s
# TPU_PEAK_TFLOPS = 197.0
# TPU_PEAK_BW = 819.0

# def get_theoretical_metrics(b, h, l, d, causal=True, dtype_bytes=2):
#     # 1. Calculate IO (Bytes)
#     #    We Read Q, K, V and Write O. Total 4 arrays of size (B, H, L, D)
#     total_elements = 4 * (b * h * l * d)
#     total_bytes = total_elements * dtype_bytes
    
#     # 2. Calculate FLOPs
#     #    Standard Attention is 4 * B * H * L * L * D
#     #    Causal masks half the matrix, so divide by 2
#     total_flops = 4 * b * h * (l * l) * d
#     if causal:
#         total_flops /= 2
        
#     return total_flops, total_bytes

# def main():
#     key = random.PRNGKey(0)
    
#     # Constants
#     BATCH = 1
#     HEADS = 16       # Use more heads to saturate the hardware
#     DIM = 128
    
#     # Block sizes (Keep constant)
#     BLOCK_Q = 1024
#     BLOCK_K = 1024
    
#     # The Sweep: Loop over Sequence Lengths
#     # Powers of 2: 1024, 2048, 4096, 8192, 16384, 32768
#     SEQ_LENS = [1024 * (2**i) for i in range(5)]
    
#     results_data = []

#     print(f"{'SeqLen':<10} | {'Time(ms)':<10} | {'TFLOP/s':<10} | {'Intensity':<10}")
#     print("-" * 55)

#     for L in SEQ_LENS:
#         # 1. Generate Inputs (New shape each time)
#         # CRITICAL: Use bfloat16 for TPU performance
#         k1, k2, k3, key = random.split(key, 4)
#         q = random.normal(k1, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)
#         k = random.normal(k2, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)
#         v = random.normal(k3, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)

#         # 2. Setup Kernel
#         mask_fn, block_mask_fn = make_causal_mask_fns(BLOCK_Q, BLOCK_K)

#         # 3. Run Benchmark
#         #    IMPORTANT: Only run "flash". "ref" will hang/crash at L=16k.
#         bench_out = benchmark.run_bench_suite(
#             q, k, v,
#             sm_scale=1.0,
#             block_b=1,
#             block_q=BLOCK_Q,
#             block_k_major=BLOCK_K,
#             block_k=512,
#             causal=False,
#             mask_fn=mask_fn,
#             block_mask_fn=block_mask_fn,
#             which=["ref"]  # <--- Only benchmark your kernel
#         )
        
#         # 4. Extract Metrics
#         #    bench_out["flash_fwd_jit"] returns (mean_time, median_time)
#         #    We use median_time to be robust against jitter
#         _, time_sec = bench_out["ref_fwd_jit"]
        
#         flops, bytes_moved = get_theoretical_metrics(BATCH, HEADS, L, DIM, causal=True)
        
#         tflops_per_sec = (flops / 1e12) / time_sec
#         intensity = flops / bytes_moved  # FLOPs per Byte

#         print(f"{L:<10} | {time_sec*1000:<10.2f} | {tflops_per_sec:<10.2f} | {intensity:<10.2f}")
        
#         results_data.append({
#             "SeqLen": L,
#             "Time_Sec": time_sec,
#             "TFLOPs": tflops_per_sec,
#             "Intensity": intensity
#         })

#     # Save to CSV for plotting
#     df = pd.DataFrame(results_data)
#     df.to_csv("roofline_data.csv", index=False)
#     print("\nSaved sweep data to roofline_data.csv")

# if __name__ == "__main__":
#     main()