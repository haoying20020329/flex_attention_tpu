from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np
import jax.numpy as jnp
# import pandas as pd  # for tabular timing results

from flex_attention_kernel import run_bench_suite as run_bench_forward
from flex_attention_dq_bwd import run_bench_suite as run_bench_dq_bwd
from flash_attention_dkv_bwd import run_bench_suite as run_bench_dkv_bwd
from mha_reference import mha_reference, mha_bwd_reference
import masks
import scores
from util import make_jax_score_fn
from constants import dimension_numbers, MIN_BLOCK_SIZE

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
    block_q_major = 1024

    # 2. Generate Inputs (BF16 for TPU Speed)
    # -------------------------
    print(f"Generating inputs: B={batch}, H={heads}, L={q_len}, D={head_dim} (BF16)...")
    k1, k2, k3, k4 = random.split(key, 4)
    key, key_do = random.split(key)

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

        # B. Prepare Forward Reference Arguments
        # Start with defaults
        forward_ref_args = {
            "sm_scale": 1.0,
            "block_b": 1,
            "block_q": block_q,
            "block_k_major": block_k_major,
            "block_k": block_k,
            "which": ["flash", "ref"], # Compare Flash vs Reference
            "mask_fn": mask_fn,
            "block_mask_fn": block_mask_fn,
        }
        # Update with case-specific args (causal=True, window_size=..., etc.)
        forward_ref_args.update(case["ref_args"])

        # C. Prepare KV Backward Reference Arguments
        # Start with defaults
        dkv_ref_args = {
            "sm_scale": 1.0,
            "block_b": 1,
            "block_q": block_q,
            "block_q_major": block_q_major,
            "block_k_major": block_k_major,
            "block_k": block_k,
            "mask_fn": mask_fn,
            "block_mask_fn": block_mask_fn
        }
        dkv_ref_args.update(case["ref_args"])

        # o, l, m = mha_reference(
        #     q.astype(jnp.float32),
        #     k.astype(jnp.float32),
        #     v.astype(jnp.float32),
        #     save_residuals=True,
        # )

        # C. Prepare KV Backward Reference Arguments
        # Start with defaults
        dq_ref_args = {
            "sm_scale": 1.0,
            "block_b": 1,
            "block_q": block_q,
            "block_k_major": block_k_major,
            "block_k": block_k,
            "mask_fn": mask_fn,
            "block_mask_fn": block_mask_fn
        }
        dq_ref_args.update(case["ref_args"])

        # C. Run Benchmarks
        # 1. forward benchmark
        forward_results, forward_outputs = run_bench_forward(
            q, k, v,
            **forward_ref_args
        )
        print("Forward summary:", forward_results)

        # 2. Get the output from the forward pass for use in backward benchmarks
        o, l, m = forward_outputs
        # Generate dO and compute the scalar "d" (di) term
        # IMPORTANT: use a PRNG key, NOT the tensor k
        do = random.normal(key_do, o.shape, dtype=jnp.bfloat16)
        d = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)
        
        # 3.Run dkv backward benchmarks using the output from the forward pass
        dkv_results = run_bench_dkv_bwd(
            q, k, v, l, m, o, do, d,
            **dkv_ref_args,
        )
        print("Backward (dkv) summary:", dkv_results)

        # 4. Run dq backward benchmarks using the output from the forward pass
        dq_results = run_bench_dq_bwd(
            q, k, v, l, m, o, do, d,
            **dq_ref_args,
        )
        print("Backward (dq) summary:", dq_results)
        
        # D. Print Quick Status
        # Check diffs in the output (if printed by benchmark)
        # Note: benchmark.py prints the diffs automatically.

    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main()




