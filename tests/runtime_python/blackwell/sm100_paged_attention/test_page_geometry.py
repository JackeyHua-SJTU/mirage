"""SM100 paged attention: the two paging geometries the online path hits.

Both cases are about how the kernel rebuilds a request's KV geometry from the
metadata `prepare_next_batch` publishes:

  seq_len      = (num_pages - 1) * PAGE_SIZE + last_page_len
  page_indices = paged_kv_indices_buffer + paged_kv_indptr[request_id]

1. PAGE BOUNDARY -- seq_len is an exact multiple of PAGE_SIZE, so last_page_len
   is PAGE_SIZE, not 0. The producer's clamp (paged_kv_last_page_len() in
   persistent_kernel.cuh) is what makes that true; the consumer must not clamp
   again. At page_size=64 with mbt=8 a prefill lands here every 8th iteration.

2. first_page_pos > 0 -- the request under test sits behind another request in
   the batch, so its page table starts at a non-zero, non-4-aligned offset into
   the shared indices buffer. A missing or mis-aligned offset reads another
   request's pages, which the disjoint page tables below turn into a hard
   mismatch rather than a plausible-looking number.

Build first:  python setup.py build_ext --inplace
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_kernel_paged_attention as rk
from pytorch_reference import paged_attention_causal_ref

NUM_KV_HEADS = 1
NUM_QO_PER_KV = 8
NUM_Q_HEADS = NUM_KV_HEADS * NUM_QO_PER_KV
HEAD_DIM = 64
PAGE_SIZE = 64
MAX_NUM_PAGES = 8
MAX_SEQ_LEN = 256
NUM_TOKENS = 8             # new tokens this call; must match MAX_TOKENS

# Pages of the request under test, and the pages of the request that precedes
# it in the batch. Disjoint, so reading the wrong window is unmistakable.
PAGE_TABLE = [5, 2, 7, 1]
LEADING_PAGES = [0, 4, 6]  # len 3 -> first_page_pos = 3, not 4-aligned

# (name, seq_len, num_leading_pages)
CASES = (
    ("page boundary, first_page_pos=0", 256, 0),
    ("mid page,      first_page_pos=3", 200, len(LEADING_PAGES)),
    ("page boundary, first_page_pos=3", 256, len(LEADING_PAGES)),
)


def gather(cache, page_table, num_rows):
    """Flatten the paged cache into contiguous [num_rows, HEAD_DIM]."""
    out = torch.empty(num_rows, HEAD_DIM, dtype=cache.dtype, device=cache.device)
    for pos in range(num_rows):
        out[pos] = cache[page_table[pos // PAGE_SIZE], pos % PAGE_SIZE, 0]
    return out


def reference(qkv, k_cache, v_cache, seq_len):
    q = qkv[:, : NUM_Q_HEADS * HEAD_DIM].view(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM)
    k_new = qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM]
    v_new = qkv[:, (NUM_Q_HEADS + 1) * HEAD_DIM :]

    prefix = seq_len - NUM_TOKENS
    k = torch.cat([gather(k_cache, PAGE_TABLE, prefix), k_new], dim=0)
    v = torch.cat([gather(v_cache, PAGE_TABLE, prefix), v_new], dim=0)

    out = paged_attention_causal_ref(q, k.unsqueeze(1), v.unsqueeze(1))
    return out.reshape(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM)


def run_case(name, seq_len, num_leading_pages, device, dtype):
    torch.manual_seed(0)
    prefix = seq_len - NUM_TOKENS
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    assert num_pages == len(PAGE_TABLE)

    # The 1-based convention the megakernel publishes: never 0, PAGE_SIZE on an
    # exact boundary. Same arithmetic as paged_kv_last_page_len().
    last_page_len = (seq_len - 1) % PAGE_SIZE + 1
    assert last_page_len == seq_len - (num_pages - 1) * PAGE_SIZE

    leading = LEADING_PAGES[:num_leading_pages]
    request_id = 1 if num_leading_pages else 0
    # Request 0 owns `leading`; request 1 (under test) owns PAGE_TABLE.
    if request_id:
        qo_indptr = [0, 0, NUM_TOKENS]
        kv_indptr = [0, num_leading_pages, num_leading_pages + num_pages]
        kv_last = [PAGE_SIZE, last_page_len]
    else:
        qo_indptr = [0, NUM_TOKENS]
        kv_indptr = [0, num_pages]
        kv_last = [last_page_len]

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    qo_indptr_t = i32(qo_indptr)
    kv_indptr_t = i32(kv_indptr)
    kv_indices_t = i32(leading + PAGE_TABLE)
    kv_last_t = i32(kv_last)

    # cos = 1, sin = 0 makes RoPE the identity.
    cos = torch.ones(MAX_SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    sin = torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    norm_w = torch.ones(HEAD_DIM, dtype=dtype, device=device)

    qkv = torch.randn(NUM_TOKENS, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                      dtype=dtype, device=device)
    k_new = qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM]
    v_new = qkv[:, (NUM_Q_HEADS + 1) * HEAD_DIM :]

    k_cache = torch.randn(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          dtype=dtype, device=device)
    v_cache = torch.randn_like(k_cache)
    k_before, v_before = k_cache.clone(), v_cache.clone()
    out = torch.zeros(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM,
                      dtype=dtype, device=device)

    rk.paged_attention_sm100(qkv, k_cache, v_cache, out, qo_indptr_t,
                             kv_indptr_t, kv_indices_t, kv_last_t, norm_w,
                             norm_w, cos, sin, 0, request_id)
    torch.cuda.synchronize()

    ok = True
    ref = reference(qkv, k_before, v_before, seq_len)
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"[{name}] seq_len={seq_len} pages={num_pages} "
          f"last_page_len={last_page_len} max |kernel - reference| = {diff:.4f}")
    if diff >= 0.05:
        print(f"[{name}] FAILED: disagrees with the reference")
        ok = False

    # The new K/V rows must land in this request's pages, at the right offsets.
    if not torch.equal(gather(k_cache, PAGE_TABLE, seq_len)[prefix:], k_new):
        print(f"[{name}] FAILED: new K rows are not in this request's pages")
        ok = False
    if not torch.equal(gather(v_cache, PAGE_TABLE, seq_len)[prefix:], v_new):
        print(f"[{name}] FAILED: new V rows are not in this request's pages")
        ok = False
    # ... and must not touch the preceding request's pages.
    for page in leading:
        if not torch.equal(k_cache[page], k_before[page]) or not torch.equal(
                v_cache[page], v_before[page]):
            print(f"[{name}] FAILED: wrote into page {page}, which belongs to "
                  f"the preceding request")
            ok = False
    return ok


def main():
    props = torch.cuda.get_device_properties(0)
    cc = props.major * 10 + props.minor
    if cc != 100:
        print(f"SKIPPED: needs SM100 (Blackwell), found cc={cc}")
        return
    device = "cuda"
    dtype = torch.bfloat16
    ok = all([run_case(name, seq_len, leading, device, dtype)
              for name, seq_len, leading in CASES])
    if not ok:
        sys.exit(1)
    print("\nPASSED: seq_len on a page boundary and a page table at a non-zero "
          "first_page_pos both resolve correctly")


if __name__ == "__main__":
    main()
