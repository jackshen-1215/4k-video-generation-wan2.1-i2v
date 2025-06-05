# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    enable_tiling_mask: bool = False,
    allowed_k_indices: torch.Tensor | None = None,
    q_tile_spatial_fg_mask: torch.Tensor | None = None,
    k_text_fg_indices: torch.Tensor | None = None,
    k_text_bg_indices: torch.Tensor | None = None,
):
    # If tile-specific masking is enabled, we must use the SDPA path.
    use_flash = (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE) and not enable_tiling_mask

    if use_flash:
        # Determine if we should bypass FlashAttention for tiling mask logic
        # This decision is now pushed into the main 'attention' function.
        if fa_version is None and enable_tiling_mask: # A simple proxy to know if we might be in tiling mask scenario
             warnings.warn(
                "Tile-specific attention masking is active; FlashAttention will be bypassed if necessary by the caller."
            ) # This warning might be too specific here, actual bypass is in `attention`

        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        # Original SDPA path if flash is not available OR if tiling mask forces it.
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        
        attn_mask_for_sdpa = None
        if enable_tiling_mask:
            _B, _num_heads, _S_q, _head_dim = q.shape 
            _S_k = k.shape[1]

            use_nuanced_mask = (
                q_tile_spatial_fg_mask is not None and 
                k_text_fg_indices is not None and 
                k_text_bg_indices is not None
            )

            if use_nuanced_mask:
                attn_mask_for_sdpa = torch.ones(_B, _S_q, _num_heads, _S_k, device=q.device, dtype=torch.bool)

                # Ensure masks/indices are on the correct device and are long tensors for indexing
                # q_tile_spatial_fg_mask is boolean, used for selecting q_indices
                q_fg_q_indices = torch.where(q_tile_spatial_fg_mask.to(device=q.device, non_blocking=True))[0]
                q_bg_q_indices = torch.where(~q_tile_spatial_fg_mask.to(device=q.device, non_blocking=True))[0]
                
                k_fg_k_indices = k_text_fg_indices.to(device=q.device, non_blocking=True, dtype=torch.long)
                k_bg_k_indices = k_text_bg_indices.to(device=q.device, non_blocking=True, dtype=torch.long)

                # Validate indices
                if k_fg_k_indices.numel() > 0 and k_fg_k_indices.max() >= _S_k:
                    raise ValueError(f"k_text_fg_indices contains an index {k_fg_k_indices.max()} out of bounds for key sequence length {_S_k}")
                if k_bg_k_indices.numel() > 0 and k_bg_k_indices.max() >= _S_k:
                    raise ValueError(f"k_text_bg_indices contains an index {k_bg_k_indices.max()} out of bounds for key sequence length {_S_k}")

                # Allow foreground Q to attend to foreground K
                if q_fg_q_indices.numel() > 0 and k_fg_k_indices.numel() > 0:
                    # Create slice for q_fg_q_indices. This is B, H, num_fg_q, S_k
                    selected_q_view_fg = attn_mask_for_sdpa.index_select(2, q_fg_q_indices)
                    # Now assign False to columns k_fg_k_indices. This is B, H, num_fg_q, num_fg_k
                    selected_q_view_fg.index_fill_(3, k_fg_k_indices, False)

                # Allow background Q to attend to background K
                if q_bg_q_indices.numel() > 0 and k_bg_k_indices.numel() > 0:
                    selected_q_view_bg = attn_mask_for_sdpa.index_select(2, q_bg_q_indices)
                    selected_q_view_bg.index_fill_(3, k_bg_k_indices, False)
            elif allowed_k_indices is not None: # Fallback to simpler allowed_k_indices mask
                attn_mask_for_sdpa = torch.ones(_B, _S_q, _num_heads, _S_k, device=q.device, dtype=torch.bool)
                if allowed_k_indices.numel() > 0:
                    k_allowed_indices = allowed_k_indices.to(device=q.device, non_blocking=True, dtype=torch.long)
                    if k_allowed_indices.max() >= _S_k:
                        raise ValueError(f"allowed_k_indices contains an index {k_allowed_indices.max()} out of bounds for key sequence length {_S_k}")
                    attn_mask_for_sdpa.index_fill_(3, k_allowed_indices, False)
            # If enable_tiling_mask is True but no specific mask info is given, attn_mask_for_sdpa remains None (full attention for the tile)

        # Transpose q, k, v for scaled_dot_product_attention
        # Input shapes: q (B, Lq, Nq, Cq), k (B, Lk, Nk, Ck), v (B, Lk, Nk, Cv)
        # Expected by SDPA: (B, N, S, E) where N=num_heads, S=seq_len, E=head_dim
        # So, q.transpose(1,2) -> (B, Nq, Lq, Cq)
        q_sdpa = q.transpose(1, 2).to(dtype)
        k_sdpa = k.transpose(1, 2).to(dtype)
        v_sdpa = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask_for_sdpa, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous() # Back to (B, Lq, Nq, Cq)
        return out
