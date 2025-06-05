Okay, let's trace how `image2video.py` (specifically, the `WanI2V.generate` method when `enable_tiling` is true) passes tile-specific information to the attention mechanism to achieve tile-specific semantic token attending.

The key components for this are:
1.  `q_tile_spatial_fg_mask`: A mask indicating which *spatial patches of the current query (Q) tile* are considered "foreground."
2.  `k_text_fg_indices`: Indices of "foreground" *text tokens in the key (K)*.
3.  `k_text_bg_indices`: Indices of "background" *text tokens in the key (K)*.
4.  `enable_tiling_mask`: A boolean flag that activates this specialized attention logic.

Here's the flow:

**1. Initialization in `image2video.py` (`WanI2V.generate`)**

*   **Global Text Token Indices (Lines ~412-423):**
    *   `k_text_fg_indices`: This is determined once per prompt, before the tiling loop begins. It's typically created as `torch.arange(text_seq_len)`, where `text_seq_len` is the original sequence length of the T5 text encoder's output (`context[0].shape[1]`). This means, by default, all text tokens from the input prompt are initially marked as "foreground."
    *   `k_text_bg_indices`: This is initialized as an empty tensor, meaning no specific text tokens are designated as "background" globally by default.
*   **Inside the Tiling Loop (per tile, Lines ~454-565):**
    *   For each tile, the code defines its spatial boundaries (`tile_top`, `tile_bottom`, `tile_left`, `tile_right`).
    *   `x_tile_model_input` (tile's latent data) and `y_tile_model_input` (tile's conditional data) are extracted.
    *   **`q_tile_spatial_fg_mask` (Lines ~482-498):** This is crucial and is created *for each tile*.
        *   It's a boolean mask with a shape corresponding to the number of spatial patches in the current tile (`num_spatial_patches_h * num_spatial_patches_w`).
        *   It identifies which of these *query spatial patches* within the tile are considered "foreground." The current implementation marks a central region of the tile (e.g., the middle 50% spatially) as foreground, and the rest as background.
    *   **`model_kwargs_tile` (Lines ~505-514):** All these pieces of information are packaged into a dictionary to be passed to the model:
        *   `'context'`: The original T5 text embeddings.
        *   `'clip_fea'`: CLIP image features.
        *   `'enable_tiling_mask': True`
        *   `'q_tile_spatial_fg_mask'`: The just-created tile-specific query foreground mask.
        *   `'k_text_fg_indices'`: The global foreground text token indices.
        *   `'k_text_bg_indices'`: The global background text token indices.
    *   This `model_kwargs_tile` is then used in the call: `noise_pred_cond_tile = self.model(x_tile_model_input, t=current_timestep_tensor, **model_kwargs_tile)[0]`. A similar call is made for `noise_pred_uncond_tile` using `context_null`.

**2. Processing in `wan/modules/model.py`**

*   **`WanModel.forward` (Lines ~600-681):**
    *   It receives all items from `model_kwargs_tile` (including `enable_tiling_mask`, `q_tile_spatial_fg_mask`, `k_text_fg_indices`, `k_text_bg_indices`).
    *   **Important Context Processing:** The input `context` (raw T5 embeddings) is passed to `self.text_embedding` (Lines ~635-639). This layer processes the text, potentially padding or truncating it to a fixed length defined by `self.text_len` (e.g., 512). The `context` tensor that gets passed to the attention blocks will have this modified text sequence length.
    *   The `clip_fea` is also processed and concatenated if it's an I2V model (Lines ~643-645). The resulting `context` passed to blocks is `[CLIP_embeddings, Processed_Text_Embeddings]`.
    *   All these kwargs are then passed down to each `WanAttentionBlock` within the main loop (`for block in self.blocks: x = block(x, **kwargs)`).
*   **`WanAttentionBlock.forward` (Lines ~341-407):**
    *   Receives these parameters.
    *   It passes `enable_tiling_mask`, `q_tile_spatial_fg_mask`, `k_text_fg_indices`, and `k_text_bg_indices` to its `self.cross_attn` module (specifically, within the `cross_attn_ffn` internal function call).
    *   For `self.self_attn`, it passes `enable_tiling_mask` and `q_tile_spatial_fg_mask`. The `k_text_..._indices` are not relevant for self-attention with `x`.
*   **`WanI2VCrossAttention.forward` (Lines ~206-272) (and similarly `WanT2VCrossAttention`):**
    *   This module handles the interaction between the video latents (query Q) and the context (key K, value V).
    *   The input `context` is split: `context_img` (from CLIP features) and `context` (the processed text embeddings).
    *   **For attention with Image Features (Lines ~245-254):**
        *   `attention` is called using `k_img`.
        *   `enable_tiling_mask` and `q_tile_spatial_fg_mask` are passed.
        *   `k_text_fg_indices` and `k_text_bg_indices` are explicitly set to `None` as this part attends to image features, not text.
        *   `allowed_k_indices` is set to `torch.arange(context_img.shape[1])` if `enable_tiling_mask` is true, meaning all query patches in the tile can attend to all image tokens.
    *   **For attention with Text Features (Lines ~257-271):**
        *   `attention` is called using the text part of `context` as K and V.
        *   `enable_tiling_mask` and `q_tile_spatial_fg_mask` are passed.
        *   Crucially, `effective_k_text_fg_indices` and `effective_k_text_bg_indices` are used. These are versions of the original `k_text_..._indices` that have been filtered to be within the bounds of the *actual sequence length of the text key tensor `k`* (e.g., 512 after processing in `WanModel.forward`). This filtering (`k_text_fg_indices[k_text_fg_indices < k.shape[1]]`) was the fix applied previously.

**3. Mask Construction in `wan/modules/attention.py` (`attention` function)**

*   **Lines ~146-250:** When `enable_tiling_mask` is `True`, the code takes the `torch.nn.functional.scaled_dot_product_attention` (SDPA) path.
*   **Mask Logic (Lines ~191-229):**
    *   An attention mask (`attn_mask_for_sdpa`) is constructed. Its shape is `(Batch, Num_Heads_Q, Seq_Len_Q, Seq_Len_K)`.
    *   If `q_tile_spatial_fg_mask`, `k_text_fg_indices`, and `k_text_bg_indices` are all present (this is the "nuanced mask" case, primarily for text attention in a tile):
        *   The mask starts by disallowing all attention (all `True`).
        *   `q_fg_q_indices`: Indices of query patches that are "foreground" within the tile (from `q_tile_spatial_fg_mask`).
        *   `q_bg_q_indices`: Indices of query patches that are "background" within the tile.
        *   The mask is then selectively set to `False` (allowing attention) as follows:
            *   Foreground Q patches (`q_fg_q_indices`) are allowed to attend to foreground K text tokens (`effective_k_text_fg_indices`).
            *   Background Q patches (`q_bg_q_indices`) are allowed to attend to background K text tokens (`effective_k_text_bg_indices`).
    *   If `allowed_k_indices` is provided (e.g., for image attention):
        *   The mask allows all Q patches to attend to the K tokens specified by `allowed_k_indices`.
*   This carefully constructed `attn_mask_for_sdpa` is then passed to `scaled_dot_product_attention`.

**In essence:**

The system allows different parts of a spatial tile (query) to focus on different semantic parts of the text prompt (key).
*   `image2video.py` defines which part of the *current tile* is considered its foreground (`q_tile_spatial_fg_mask`).
*   It also (globally) defines which *text tokens* are foreground/background (`k_text_fg_indices`, `k_text_bg_indices`).
*   This information is threaded through the model layers to the `attention.py` module.
*   `attention.py` builds a specific mask so that:
    *   The "foreground" query patches of a tile primarily attend to "foreground" text tokens.
    *   The "background" query patches of a tile primarily attend to "background" text tokens (if any are defined and available).
    *   For image features, the mechanism is simpler: all parts of the query tile can attend to all image features.

This enables the model to, for example, render the main subject (foreground of the tile) with details from the main parts of the prompt (foreground text tokens), while peripheral areas of the tile might draw from more general or background elements of the prompt.