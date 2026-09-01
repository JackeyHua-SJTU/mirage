"""Shared PyTorch reference for the SM100 paged-attention tests."""

import torch


def paged_attention_causal_ref(q, k, v, window_size=0):
    """Causal (optionally sliding-window) attention, computed in float32.

    q:    [num_tokens, num_q_heads, head_dim] -- this call's NEW tokens
    k, v: [seq_len,    num_kv_heads, head_dim] -- cached prefix + the new tokens
    Query t sits at absolute position ``seq_len - num_tokens + t``, so the mask
    is the same one the kernel builds from seq_len. GQA follows the kernel's
    layout: ``num_q_heads // num_kv_heads`` consecutive q heads per kv head.

    Returns [num_tokens, num_q_heads, head_dim] in q's dtype.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    seq_len, num_kv_heads, _ = k.shape
    group = num_q_heads // num_kv_heads
    prefix = seq_len - num_tokens

    k_f = k.float().repeat_interleave(group, dim=1)
    v_f = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->ths", q.float(), k_f) / head_dim ** 0.5

    key_pos = torch.arange(seq_len, device=q.device)
    query_pos = torch.arange(prefix, seq_len, device=q.device)
    keep = key_pos[None, :] <= query_pos[:, None]
    if window_size > 0:
        keep &= key_pos[None, :] > query_pos[:, None] - window_size

    scores = scores.masked_fill(~keep[:, None, :], float("-inf"))
    out = torch.einsum("ths,shd->thd", torch.softmax(scores, dim=-1), v_f)
    return out.to(q.dtype)
