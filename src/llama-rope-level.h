#pragma once

#include "ggml.h"

#include <cmath>

// elastic YaRN: one set of rope parameters, see llama_cparams::rope_levels
struct llama_rope_params {
    float freq_scale;
    float ext_factor;
    float attn_factor;
};

// move keys roped with `from` at positions pos to `to` at the same positions:
// rotating by -pos (pos_neg) undoes the rotation, as it is orthogonal; its magnitude attn_factor*m, where ggml_rope
// adds m = 1 + 0.1*ln(1/freq_scale) for YaRN (ext_factor != 0), is divided out by passing 1/(attn_factor*m*m)
// x: F32 [n_embd_head, n_head, n], sections: nullptr for a plain (non M-RoPE) mode
static inline ggml_tensor * llama_rope_level_move(
        ggml_context            * ctx,
        ggml_tensor             * x,
        ggml_tensor             * pos,
        ggml_tensor             * pos_neg,
        ggml_tensor             * factors,
        int                       n_rot,
        int                     * sections,
        int                       mode,
        int                       n_ctx_orig,
        float                     freq_base,
        const llama_rope_params & from,
        const llama_rope_params & to,
        float                     beta_fast,
        float                     beta_slow) {
    const float m_from     = from.ext_factor != 0.0f ? 1.0f + 0.1f*logf(1.0f/from.freq_scale) : 1.0f;
    const float unrot_attn = 1.0f/(from.attn_factor*m_from*m_from);

    if (sections) {
        x = ggml_rope_multi(ctx, x, pos_neg, factors, n_rot, sections, mode, n_ctx_orig,
                freq_base, from.freq_scale, from.ext_factor, unrot_attn, beta_fast, beta_slow);
        x = ggml_rope_multi(ctx, x, pos, factors, n_rot, sections, mode, n_ctx_orig,
                freq_base, to.freq_scale, to.ext_factor, to.attn_factor, beta_fast, beta_slow);
    } else {
        x = ggml_rope_ext(ctx, x, pos_neg, factors, n_rot, mode, n_ctx_orig,
                freq_base, from.freq_scale, from.ext_factor, unrot_attn, beta_fast, beta_slow);
        x = ggml_rope_ext(ctx, x, pos, factors, n_rot, mode, n_ctx_orig,
                freq_base, to.freq_scale, to.ext_factor, to.attn_factor, beta_fast, beta_slow);
    }

    return x;
}
