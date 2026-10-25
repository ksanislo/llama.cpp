#pragma once

// Prompt pipe (port of Strata's --prefill-pipe, Prefill::pipeline_chunk): with pipeline parallelism over S layer-split
// devices, a prompt of n tokens read in chunks of C tokens costs about sum_i (b + a C_i) + (S - 1) max_i (b + a C_i):
// b is the per-chunk pass over a stage's expert weights (uploaded where they are not in VRAM), a the per-token work.
// In units of a, with r = b/a: k r + n + (S - 1)(r + C) for k chunks of C tokens (the last one the rest).
// The rule tries every chunk count from the fewest the buffers allow, each chunk evened out on the 256-token grid and
// at least 512 tokens, and keeps the cheapest when it beats chunks of `cap` by 3% (integer arithmetic: no near-tie
// can fall either way). r <= 1 (`1`): today's chunk count evened out, only when the last chunk has at least
// stream_min tokens (a short tail is nearly free; evening it out adds a full pass over the experts).

#include <algorithm>
#include <cmath>
#include <cstdint>

static inline int64_t llama_prompt_pipe_ratio_tokens(double ratio) {
    return ratio > 0.0 ? (int64_t) std::llround(std::min(ratio, 1048576.0)) : 0;
}

// the chunk size for a batch of n tokens; returns cap when the plan does not change it
static inline int64_t llama_prompt_pipe_chunk(int64_t n, int64_t cap, int stages, double ratio, int64_t stream_min = 1024) {
    if (!(ratio > 0.0) || stages < 2 || n <= 0 || cap <= 0) {
        return cap;
    }
    const int64_t r  = ratio > 1.0 ? llama_prompt_pipe_ratio_tokens(ratio) : 0;
    const int64_t S1 = stages - 1;
    if (r == 0) {
        const int64_t k    = (n + cap - 1) / cap;
        const int64_t last = n - (k - 1) * cap;
        if (k <= 1 || last < stream_min) {
            return cap;
        }
        const int64_t per = (n + k - 1) / k;
        return std::min(cap, ((per + 255) / 256) * 256);
    }
    auto cost = [&](int64_t c) {
        const int64_t k = (n + c - 1) / c;
        return k * r + n + S1 * (r + std::min(c, n));
    };
    const int64_t today      = std::min(n, cap);
    const int64_t today_cost = cost(today);
    const int64_t k_min      = (n + cap - 1) / cap;
    const int64_t k_max      = std::max<int64_t>(k_min, std::min<int64_t>(64, (n + 511) / 512));
    int64_t best      = today;
    int64_t best_cost = today_cost;
    for (int64_t k = k_min; k <= k_max; ++k) {
        const int64_t c = std::min(cap, std::max<int64_t>(512, (((n + k - 1) / k + 255) / 256) * 256));
        const int64_t v = cost(c);
        if (v < best_cost) {
            best_cost = v;
            best      = c;
        }
    }
    return (best < today && 100 * best_cost < 97 * today_cost) ? best : cap;
}
