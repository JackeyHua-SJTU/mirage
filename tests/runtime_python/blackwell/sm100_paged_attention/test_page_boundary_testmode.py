"""Test-mode test: a prefill whose KV length lands exactly on a page boundary.

Runs the real `prepare_next_batch` -> `multitoken_paged_attention_sm100` chain
with page_size=64 and a 128-token prefill, so the request ends on an exact page
multiple. `paged_kv_last_page_len` is 1-based, so the metadata must read 64 (a
full last page), never 0 -- attention rebuilds

    seq_len = (num_pages - 1) * page_size + last_page_len

and a 0 there silently drops a whole page, which at page_size=64 and mbt=8
happens on every eighth prefill iteration of the online path.

The test asserts both halves: the published metadata, and the attention output
against the PyTorch reference.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import paged_attention_causal_ref

HEAD_DIM = 64
NUM_Q_HEADS = 1
NUM_KV_HEADS = 1
PAGE_SIZE = 64
MAX_NUM_PAGES = 4
MAX_SEQ_LENGTH = 256
T = 2 * PAGE_SIZE  # prefill length == max_num_batched_tokens; 2 whole pages
FUSED_DIM = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM


def require_sm100():
    props = torch.cuda.get_device_properties(0)
    cc = props.major * 10 + props.minor
    if cc != 100:
        print(f"SKIPPED: needs SM100 (Blackwell), found cc={cc}")
        return False
    return True


def main():
    if not require_sm100():
        return
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["max_num_batched_tokens"] = T
    params["max_num_batched_requests"] = 1
    params["page_size"] = PAGE_SIZE
    params["max_num_pages"] = MAX_NUM_PAGES
    params["max_seq_length"] = MAX_SEQ_LENGTH
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor([T], dtype=torch.int32, device=device),
    }
    pk = PersistentKernel(**params)

    q = torch.randn(T, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(T, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(T, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    qkv = torch.cat([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)],
                    dim=1).contiguous()

    # cos = 1, sin = 0 makes RoPE the identity, so the reference is plain
    # causal attention and any mismatch points at the paging geometry.
    cos = torch.ones(MAX_SEQ_LENGTH, HEAD_DIM, dtype=dtype, device=device)
    sin = torch.zeros(MAX_SEQ_LENGTH, HEAD_DIM, dtype=dtype, device=device)
    # qk-norm is off; these are the dummies the layer API still requires.
    q_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    k_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    k_cache = torch.zeros(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    out = torch.zeros(T, NUM_Q_HEADS * HEAD_DIM, dtype=dtype, device=device)

    dts = {}
    for name, tensor in (("qkv", qkv), ("k_cache", k_cache),
                         ("v_cache", v_cache), ("qn", q_norm), ("kn", k_norm),
                         ("cos", cos), ("sin", sin), ("out", out)):
        dts[name] = pk.attach_input(tensor, name=name)

    pk.paged_attention_layer(
        input=dts["qkv"], k_cache=dts["k_cache"], v_cache=dts["v_cache"],
        q_norm=dts["qn"], k_norm=dts["kn"],
        cos_pos_embed=dts["cos"], sin_pos_embed=dts["sin"],
        output=dts["out"],
        grid_dim=(1, NUM_KV_HEADS, 1), block_dim=(128, 1, 1),
        enable_qk_norm=False,
    )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    # The indptrs are reset by the second (finalizing) prepare_next_batch pass,
    # but last_page_len is only ever written for live slots, so slot 0 still
    # holds what the prefill iteration published.
    last_page_len = int(pk.meta_tensors["paged_kv_last_page_len_buffer"][0])
    print(f"published last_page_len={last_page_len} for a {T}-token prefill "
          f"at page_size={PAGE_SIZE}")
    if last_page_len != PAGE_SIZE:
        print(f"FAILED: last_page_len is {last_page_len}, expected {PAGE_SIZE} "
              f"-- a page boundary must report a FULL last page, never 0")
        ok = False

    ref = paged_attention_causal_ref(q, k, v).reshape(T, NUM_Q_HEADS * HEAD_DIM)
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"out max diff: {diff:.3e}")
    try:
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    except AssertionError as e:
        print(f"FAILED: {e}")
        ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("PASSED: a page-aligned prefill keeps its last page")


if __name__ == "__main__":
    main()
