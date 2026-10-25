// LLAMA_PROMPT_PIPE's chunk planner against the plans Strata's --prefill-pipe logged on the same rig (4 stages)
#include "../src/llama-prompt-pipe.h"

#include <cstdio>

int main() {
    struct tcase { int64_t n, cap; int stages; double r; int64_t want; };
    const tcase cases[] = {
        // Strata Qwen3.8-Flash-Next UD-Q4_K_XL (--prefill-pipe 1536), chunk cap 8192: "N tokens in C-token chunks over 4 stages"
        { 43445, 8192, 4, 1536.0,  4864 },
        {  4096, 8192, 4, 1536.0,  2048 },
        {  1731, 8192, 4, 1536.0,  1024 },
        // Strata Q8_0 (--prefill-pipe 6144)
        {  5120, 8192, 4, 6144.0,  2560 },
        // no pipeline: one stage, or no ratio
        { 21054, 8192, 1, 1536.0,  8192 },
        { 21054, 8192, 4,    0.0,  8192 },
        // `1`: today's chunk count evened out on the 256 grid, only when the last chunk is >= 1024 tokens
        { 21054, 8192, 4,    1.0,  7168 },
        { 16400, 8192, 4,    1.0,  8192 },
        {  4000, 8192, 4,    1.0,  8192 },
        // a short prompt is not split below 512 tokens a chunk
        {   600, 8192, 4, 1536.0,  8192 },
    };
    int fails = 0;
    for (const tcase & c : cases) {
        const int64_t got = llama_prompt_pipe_chunk(c.n, c.cap, c.stages, c.r);
        const bool ok = got == c.want;
        fails += !ok;
        printf("%s n=%lld cap=%lld S=%d r=%g -> %lld (want %lld)\n", ok ? "OK  " : "FAIL",
               (long long) c.n, (long long) c.cap, c.stages, c.r, (long long) got, (long long) c.want);
    }
    // invariants over a grid: within [512, cap] when changed, on the 256 grid, never more than 64 chunks
    for (int64_t n = 1; n <= 70000; n += 37) {
        for (const double r : { 1.0, 768.0, 1536.0, 2700.0, 6144.0 }) {
            const int64_t c = llama_prompt_pipe_chunk(n, 8192, 4, r);
            if (c != 8192 && (c < 512 || c > 8192 || c % 256 != 0 || (n + c - 1)/c > 64)) {
                printf("FAIL n=%lld r=%g -> %lld\n", (long long) n, r, (long long) c);
                fails++;
            }
        }
    }
    printf("%s\n", fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
