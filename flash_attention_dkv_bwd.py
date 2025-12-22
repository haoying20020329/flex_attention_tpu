#@title Imports
import functools
from typing import Callable

import jax
import jax.experimental.pallas
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np
import time
import statistics as stats
from jax import lax 
from flex_attention_kernel import _flex_attention_impl
from mha_reference import mha_reference,mha_bwd_reference
from util import make_jax_score_fn
import scores
import masks
import pandas as pd

dimension_numbers = (((1,), (1,)), ((), ()))
MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((0,), (0,)), ((), ()))



def flash_attention_bwd_dkv(
    k,
    v,
    q,
    ab,
    l,
    m,
    di,
    do,
    *,
    block_q,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    # mask_value: float,
    debug: bool,
    score_fn = None,
    mask_fn = None,      # Added
    block_mask_fn = None # Added
):
    batch_size, head_num, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape

    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

    grid = (
      batch_size,
      head_num,
      kv_seq_len // block_k_major,
      q_seq_len // block_q_major,
    )



    def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
      return (batch_index, head_index, q_seq_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
    do_spec = qo_spec
    assert do.ndim == len(qo_spec.block_shape)

    def kv_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
      if block_mask_fn is not None:
        next_kv_index = jax.lax.select(
            block_mask_fn(q_seq_index, kv_seq_index),
            kv_seq_index,
            0,
        )
      else:
        next_kv_index = kv_seq_index
      return (batch_index, head_index, next_kv_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim),  kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
      return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
        return (batch_index, head_index, q_seq_index, kv_seq_index)

    dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
    )
    
    def dkv_index_map(batch_index, head_index, kv_seq_index, _):
      return (batch_index, head_index, kv_seq_index, 0)
  
    dk_spec = kv_spec
    dv_spec = kv_spec

    # in_spec specify
    in_specs = [
      qo_spec,      # q
      kv_spec,      # k
      kv_spec,      # v
      dab_spec,     # bias
      lm_spec,      # l
      lm_spec,      # m
      di_spec,      # di
      qo_spec,      # do   
    ]

    out_shapes = [
      jax.ShapeDtypeStruct(k.shape, k.dtype),
      jax.ShapeDtypeStruct(v.shape, v.dtype),
      jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
    ]
    
    out_specs = [
        dk_spec,
        dv_spec,
        dab_spec,
    ]

    # Allocate scratch buffers
    # if block_k != kv_seq_len:
    dk_scratch = pltpu.VMEM((block_k_major, head_dim), jnp.float32)
    dv_scratch = pltpu.VMEM((block_k_major, head_dim), jnp.float32)
    # scratch_shapes = [dq_scratch, dk_scratch, dv_scratch]
    scratch_shapes = [dk_scratch, dv_scratch]
    # else:
    #     scratch_shapes = []

    kernel = functools.partial(
        flash_attention_dkv_kernel,
        causal = causal,
        sm_scale = sm_scale,
        block_q = block_q,
        block_k = block_k,
        q_seq_len = q_seq_len,
        block_q_major = block_q_major,
        block_k_major=block_k_major,
        score_fn= score_fn,
        mask_fn=mask_fn,               
        block_mask_fn=block_mask_fn    
    )

    # dq = jnp.zeros(dq_spec, jnp.float32)

    dk_out, dv_out, *aux = pl.pallas_call(
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
  )(q, k, v, ab, l, m, di, do)

    return dk_out, dv_out
    

def flash_attention_dkv_kernel(
      q_tile_ref,
      k_tile_ref,
      v_tile_ref,
      ab_tile_ref,
      l_tile_ref,
      m_tile_ref,
      di_tile_ref,
      dO_tile_ref,
      dk_tile_ref,
      dv_tile_ref,
      dab_tile_ref,
      dk_scratch_ref, 
      dv_scratch_ref,
      *,
      causal,
      sm_scale,
      block_q,
      block_k,
      q_seq_len,
      block_q_major,
      block_k_major,
      score_fn,
      mask_fn,
      block_mask_fn
  ):
    jax.experimental.pallas.debug_print("nothing")

    _, _, q_seq_length, _ = q_tile_ref.shape
    # Grid: (Batch, Head, Major_K, Major_Q)
    # program_id(2) -> Major K Index
    # program_id(3) -> Major Q Index
    kv_tile_idx = pl.program_id(axis = 2)
    q_tile_idx = pl.program_id(axis = 3)
    
    # Initialization of scratch buffers
    @pl.when(q_tile_idx == 0)
    def start_new_kv_seq():
        dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
        dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

    # --- BLOCK MASK CHECK (OUTER) ---
    # jax.debug.print("checkpointing")
    # jax.experimental.pallas.debug_print("kv {}, q {}",kv_tile_idx.astype(jnp.int32), q_tile_idx.astype(jnp.int32))
    
    if block_mask_fn is None:
      should_run = True
    else:
      # We check once for the entire Major Q / Major K block pair
      should_run = block_mask_fn(q_tile_idx, kv_tile_idx)

      
    @pl.when(should_run)
    def body():
      @pl.loop(0, block_q_major // block_q, unroll=True)
      def _body(j):
          start_q = j * block_q
          
          @pl.loop(0, block_k_major // block_k, unroll=True)
          def _body(i):
            start_k = i * block_k
            
            # Load Data
            q  = q_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            dO = dO_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            di  = di_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            l  = l_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            m  = m_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            k = k_tile_ref[0, 0, pl.dslice(start_k, block_k), :].astype(jnp.float32)
            v = v_tile_ref[0, 0, pl.dslice(start_k, block_k), :].astype(jnp.float32)
            
            dk_past = dk_scratch_ref[pl.ds(start_k, block_k), :]
            dv_past = dv_scratch_ref[pl.ds(start_k, block_k), :]
                        
            # Forward Pass Recomputation
            if score_fn is not None:
                S, score_grad_fn = jax.vjp(score_fn, q, k)
            else:
                S = jax.lax.dot_general(q, k, dimension_numbers, preferred_element_type=jnp.float32)
                score_grad_fn = None

            S = S * sm_scale
            
            if ab_tile_ref is not None:
                ab = ab_tile_ref[0, 0, pl.dslice(start_q, block_q), pl.dslice(start_k, block_k)]
                S += ab.astype(jnp.float32)

            # --- MASK APPLICATION (INNER) ---
            if mask_fn is not None:
                # Calculate absolute coordinates
                q_off = q_tile_idx * block_q_major + start_q
                k_off = kv_tile_idx * block_k_major + start_k
                
                q_pos = q_off + jnp.arange(block_q, dtype=jnp.int32)
                k_pos = k_off + jnp.arange(block_k, dtype=jnp.int32)
                
                # Apply mask to Scores (S) before softmax
                # S = S + mask_val
                token_mask = mask_fn(q_pos, k_pos)
                S = S + jnp.where(token_mask, 0.0, -0.7 * float(jnp.finfo(jnp.dtype("float32")).max))

            unnormalized = jnp.exp(S - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1))   
            P = unnormalized / pltpu.repeat(l, block_k // MIN_BLOCK_SIZE, axis=1)                 

            # Backward: dV
            dv = dv_past + jax.lax.dot_general(P, dO, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)
            dv_scratch_ref[pl.dslice(start_k, block_k), :] = dv.astype(dv_scratch_ref.dtype)

            # Backward: dP
            dP = jax.lax.dot_general(dO, v, dimension_numbers, preferred_element_type=jnp.float32)
  
            # Backward: dS
            dS = P * (dP - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1))

            if sm_scale != 1.0:
              dS = dS * sm_scale  

            # Backward: dQ, dK
            if score_grad_fn is not None:
              dq_update, dk_update = score_grad_fn(dS)
            else:
              dk_update = jax.lax.dot(dS.T.astype(dO.dtype), q, preferred_element_type=jnp.float32)
            
            dk = dk_past + dk_update
            dk_scratch_ref[pl.dslice(start_k, block_k), :] = dk.astype(dk_scratch_ref.dtype)

    # Store results when we finish the last Q block for this K block
    @pl.when(q_tile_idx == q_seq_len // block_q_major - 1)
    def store_res():
      dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
      dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)
    # return flash_attention_dkv_kernel


def flop_count_attention(b, h, q, k, d):
    """
    Rough FLOP count for one forward pass of scaled dot-product attention:
      QK^T  : 2 * b * h * q * k * d        (matrix multiplication)
      Softmax: ~ b * h * q * k             (small, we ignore it)
      (softmax @ V): 2 * b * h * q * k * d (another matmul)
    Total ≈ 4 * b * h * q * k * d FLOPs
    """
    return 4.0 * b * h * q * k * d

def benchmark(fn, args, iters=1, warmup=1, name="fn"):
    # 1. Warmup phase — triggers JIT compilation and stabilizes cache
    for _ in range(warmup):
        y = fn(*args)
        # .block_until_ready() ensures we wait until computation is finished
        if isinstance(y, (tuple, list)):
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, y
            )
        else:
            y.block_until_ready()
        jax.effects_barrier()

    # 2. Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn(*args)
        # Synchronize (very important for accurate timing)
        if isinstance(y, (tuple, list)):
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, y
            )
        else:
            y.block_until_ready()
        jax.effects_barrier()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # 3. Compute summary statistics
    mean_t = sum(times) / len(times)
    med_t = stats.median(times)
    p10, p90 = np.percentile(np.array(times), [10, 90])

    print(f"[{name}] mean={mean_t*1e3:.2f} ms  median={med_t*1e3:.2f} ms  "
          f"p10={p10*1e3:.2f} ms  p90={p90*1e3:.2f} ms")

    # Return average and median latency (seconds)
    return mean_t, med_t

def build_fns_for_bench(
    q, k, v,
    l, m, o, do, di,
    *,
    ab=None,
    sm_scale=1.0,
    causal=False,
    block_b=1,
    block_q=128,
    block_k_major=128,
    block_k=128,
    debug=False,
    block_q_major=128,
    score_fn=None,
    window_size=None,
    segment_ids=None,
    s2_stride=None,
    alibi_slope=None,
    mask_fn=None,
    block_mask_fn=None,
):
    """
    Build JIT-ed reference and Pallas (flex) dK/dV backward fns.

    - Reference: mha_bwd_reference(q, k, v, o, do, l, m, ...)
      returns (dq_ref, dk_ref, dv_ref)
    - Flex (Pallas): flash_attention_bwd_dkv(q, k, v, l, m, di, do, ...)
      returns (dk_flex, dv_flex) or a tuple whose first two are dk, dv.
    """

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

    # ----------------- FlexAttention backward (Pallas dK/dV) -----------------
    # We treat the Pallas dK/dV kernel as "flex"
    flex_partial = functools.partial(
        flash_attention_bwd_dkv,
        ab=ab,
        causal=causal,
        sm_scale=sm_scale,
        block_q=block_q,
        block_q_major=block_q_major,
        block_k_major=block_k_major,
        block_k=block_k,
        debug=debug,
        score_fn=score_fn,
        mask_fn=mask_fn,
        block_mask_fn=block_mask_fn,
    )

    def flex_bwd_fn(q, k, v, l, m, di, do):
        out = flex_partial(q=q, k=k, v=v, l=l, m=m, di=di, do=do)
        # Normalize outputs to (dk, dv)
        if isinstance(out, (tuple, list)):
            # Assume (dk, dv, *rest) or just (dk, dv)
            return out[0], out[1]
        # If kernel returns a single packed tensor, you can change this as needed
        return out

    flex_jit = jax.jit(flex_bwd_fn)

    return ref_jit, flex_jit


def run_bench_suite(
    q, k, v, l, m, o, do, di,
    *,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_q_major,
    block_k,
    causal=False,
    score_fn=None,
    window_size=None,
    segment_ids=None,
    s2_stride=None,
    alibi_slope=None,
    mask_fn=None,
    block_mask_fn=None,
    ab=None,
    debug=False,
):
    b, h, q_len, h_d = q.shape
    _, _, k_len, _ = k.shape

    gflops = flop_count_attention(b, h, q_len, k_len, h_d) / 1e9

    ref_jit, flex_jit = build_fns_for_bench(
        q, k, v,
        l, m, o, do, di,
        ab=ab,
        sm_scale=sm_scale,
        causal=causal,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        block_q_major=block_q_major,
        debug=debug,
        score_fn=score_fn,
        window_size=window_size,
        segment_ids=segment_ids,
        s2_stride=s2_stride,
        alibi_slope=alibi_slope,
        mask_fn=mask_fn,
        block_mask_fn=block_mask_fn,
    )

    print(f"\n== Benchmark config (dK/dV): "
          f"b={b}, h={h}, q={q_len}, k={k_len}, d={h_d}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs")

    # ----------------- Reference timing -----------------
    t_mean_ref, t_med_ref = benchmark(
        ref_jit,
        (q, k, v, o, do, l, m),          # (q, k, v, o, do, l, m)
        name="mha_reference_bwd_dkv[jit]",
    )
    print(f"  → Reference throughput: {gflops / t_med_ref:.2f} GFLOP/s")

    # ----------------- Flex (Pallas dK/dV) timing -----------------
    t_mean_flex, t_med_flex = benchmark(
        flex_jit,
        (q, k, v, l, m, di, do),         # (q, k, v, l, m, di, do)
        name="flex_bwd_dkv[jit]",
    )
    print(f"  → Flex throughput:      {gflops / t_med_flex:.2f} GFLOP/s")

    # ----------------- Numeric correctness -----------------
    dq_ref, dk_ref, dv_ref = ref_jit(q, k, v, o, do, l, m)
    dk_ref = dk_ref.block_until_ready()
    dv_ref = dv_ref.block_until_ready()

    dk_flex, dv_flex = flex_jit(q, k, v, l, m, di, do)
    dk_flex = dk_flex.block_until_ready()
    dv_flex = dv_flex.block_until_ready()
    jax.effects_barrier()

    rel_err_dk = jnp.linalg.norm(dk_flex - dk_ref) / (jnp.linalg.norm(dk_ref) + 1e-6)
    rel_err_dv = jnp.linalg.norm(dv_flex - dv_ref) / (jnp.linalg.norm(dv_ref) + 1e-6)

    rel_err_dk_f = float(rel_err_dk)
    rel_err_dv_f = float(rel_err_dv)

    print(f"Numeric diff Flex vs Ref (Relative L2) dK: {rel_err_dk_f:.3e}")
    print(f"Numeric diff Flex vs Ref (Relative L2) dV: {rel_err_dv_f:.3e}")

    return {
        "ref_ms_med": t_med_ref * 1e3,
        "flex_ms_med": t_med_flex * 1e3,
        "ref_gflops": gflops / t_med_ref,
        "flex_gflops": gflops / t_med_flex,
        "rel_l2_dk": rel_err_dk_f,
        "rel_l2_dv": rel_err_dv_f,
    }
    


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

def main():
    print("=== FlexAttention dK/dV Backward Benchmark & Verification ===\n")

    # 1. Hardware / Data Config
    # -------------------------
    key = random.PRNGKey(0)
    batch, heads = 1, 8
    q_len, kv_len = 4096, 4096
    head_dim = 128

    # Block sizes (must match your dkv bwd kernel tiling)
    block_b = 1
    block_q = 1024           # used by dkv kernel
    block_q_major = 1024     # if your impl distinguishes "major" q-blocks
    block_k_major = 1024
    block_k = 1024

    # 2. Generate Inputs (BF16 for TPU speed)
    # -------------------------
    print(f"Generating inputs: B={batch}, H={heads}, L={q_len}, D={head_dim} (BF16)...")
    k1, k2, k3, k4, key = random.split(key, 5)
    q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.bfloat16)
    k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.bfloat16)
    v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.bfloat16)
    do = random.normal(k4, (batch, heads, q_len, head_dim), dtype=jnp.bfloat16)

    ab = None
    segment_ids = None
    sm_scale = 1.0

    # 3. Define Test Cases (same semantics as dq main)
    # -------------------------
    test_cases = []

    # test_cases.append({
    #     "name": "Causal Attention",
    #     "factory": masks.make_causal_mask_fns,
    #     "factory_args": (),   # no extra args
    #     "ref_args": {"causal": True},
    # })

    # --- Case B: Sliding Window ---
    window_size = 2048
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
    # alibi_fn = make_jax_score_fn(
    #     scores.make_alibi_score_fn(slope=0.5)
    # )
    # test_cases.append({
    #     "name": "ALiBi Attention",
    #     "factory": lambda *args: (None, None),  # no masks: score_fn only
    #     "factory_args": (),
    #     "ref_args": {
    #         "score_fn": alibi_fn,
    #         "alibi_slope": 0.5,
    #         "causal": False,
    #     },
    # })

    # --- Case E: Tanh Soft-Capping (Score Function) ---
    # tanh_fn = make_jax_score_fn(
    #     scores.make_softcap_score_fn(cap=30.0)
    # )
    # test_cases.append({
    #     "name": "Tanh Soft-Capping",
    #     "factory": lambda *args: (None, None),
    #     "factory_args": (),
    #     "ref_args": {
    #         "score_fn": tanh_fn,
    #         "causal": False,
    #     },
    # })

    # (If you later want causal / sliding / jagged, you can reuse the same
    # factories & ref_args pattern from the dq main and just extend test_cases.)

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
            "block_q_major": block_q_major,
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

# def main():
#     key = random.PRNGKey(0)

#     # Constants
#     BATCH = 1
#     HEADS = 8
#     DIM = 128

#     # Block sizes (keep constant)
#     BLOCK_Q = 1024
#     BLOCK_K_MAJOR = 1024
#     BLOCK_K = 1024

#     # Sequence length sweep (same style as fwd)
#     SEQ_LENS = [1024 * (2 ** i) for i in range(5)]  # 1k, 2k, 4k, 8k, 16k

#     results_data = []

#     print(f"{'SeqLen':<10} | {'Time(ms)':<10} | {'TFLOP/s':<10} | {'Intensity':<10}")
#     print("-" * 55)

#     def my_score(q, k):
#         # Tile-local score; wrapped by make_jax_score_fn
#         return jnp.einsum("qd,kd->qk", q, k)

#     for L in SEQ_LENS:
#         # 1. Generate inputs
#         k1, k2, k3, k4, key = random.split(key, 5)

#         # Use bf16 for inputs to match TPU execution mode
#         q = random.normal(k1, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)
#         k = random.normal(k2, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)
#         v = random.normal(k3, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)

#         # 2. Forward pass (reference) to get O, l, m
#         sm_scale = 1.0
#         o, l, m = mha_reference(
#             q.astype(jnp.float32),
#             k.astype(jnp.float32),
#             v.astype(jnp.float32),
#             sm_scale=sm_scale,
#             save_residuals=True,
#         )

#         # 3. Generate dO and compute the scalar "d" (di) term
#         do = random.normal(k4, o.shape, dtype=jnp.bfloat16)
#         d = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)

#         # 4. Build mask + score functions
#         # mask_fn, block_mask_fn = make_causal_mask_fns(
#         #     block_q=BLOCK_Q, block_k_major=BLOCK_K_MAJOR
#         # )
#         jax_score = make_jax_score_fn(my_score)

#         # 5. Jitted backward kernel
#         def bwd_call(q_, k_, v_, o_, l_, m_, do_, d_):
#             return flash_attention_bwd_dkv(
#                 q=q_, k=k_, v=v_,
#                 ab=None,
#                 l=l_, m=m_,
#                 do=do_,di=d_,
#                 block_q=BLOCK_Q,
#                 block_k_major=BLOCK_K_MAJOR,
#                 block_k=BLOCK_K,
#                 block_q_major=BLOCK_Q,
#                 sm_scale=sm_scale,
#                 debug=False,
#                 score_fn=None,
#                 mask_fn=None,
#                 block_mask_fn=None,
#                 causal=False
#             )

#         bwd_jit = jax.jit(bwd_call)

#         # 6. Benchmark backward kernel only
#         _, time_sec = benchmark(
#             bwd_jit,
#             (q, k, v, o, l, m, do, d),
#             name=f"flex_bwd_dq_L{L}",
#         )

#         # 7. Theoretical FLOPs and bytes for backward
#         flops, bytes_moved = get_theoretical_metrics_bwd(
#             BATCH, HEADS, L, DIM, causal=True, dtype_bytes=2
#         )

#         tflops_per_sec = (flops / 1e12) / time_sec
#         intensity = flops / bytes_moved

#         print(f"{L:<10} | {time_sec * 1e3:<10.2f} | {tflops_per_sec:<10.2f} | {intensity:<10.2f}")

#         results_data.append({
#             "SeqLen": L,
#             "Time_Sec": float(time_sec),
#             "TFLOPs": float(tflops_per_sec),
#             "Intensity": float(intensity),
#         })

#     # 8. Save CSV for backward roofline plot
#     df = pd.DataFrame(results_data)
#     df.to_csv("roofline_data_bwd_dkv.csv", index=False)
#     print("\nSaved backward sweep data to roofline_data_bwd.csv")

# if __name__ == "__main__":
#     main()
