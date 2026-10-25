// elastic YaRN: moving cached keys between rope levels (llama_rope_level_move, as llama_kv_cache::build_graph_rope_level
// does it on a gathered set of cache rows) must match roping them with the new level directly, for the M-RoPE modes
// (qwen4exp ropes IMROPE) and the plain ones; runs on every backend, CUDA_VISIBLE_DEVICES= keeps it on the CPU

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../src/llama-rope-level.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const llama_rope_params plain = { 1.0f,  0.0f, 1.0f };
static const llama_rope_params yarn  = { 0.25f, 1.0f, 1.0f };
static const llama_rope_params yarn2 = { 0.25f, 1.0f, 0.9f }; // a non-trivial magnitude

static const int   n_ctx_orig = 262144;
static const float freq_base  = 10000000.0f;

struct test_case {
    const char * name;
    int  mode;
    int  n_embd_head;
    int  n_rot;
    int  sections[4];
    bool mrope;
};

static ggml_tensor * rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos, const test_case & tc, const llama_rope_params & p) {
    int sections[4];
    memcpy(sections, tc.sections, sizeof(sections));

    if (tc.mrope) {
        return ggml_rope_multi(ctx, x, pos, nullptr, tc.n_rot, sections, tc.mode, n_ctx_orig, freq_base,
                p.freq_scale, p.ext_factor, p.attn_factor, 32.0f, 1.0f);
    }

    return ggml_rope_ext(ctx, x, pos, nullptr, tc.n_rot, tc.mode, n_ctx_orig, freq_base,
            p.freq_scale, p.ext_factor, p.attn_factor, 32.0f, 1.0f);
}

static ggml_tensor * move(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos, ggml_tensor * pos_neg, const test_case & tc,
        const llama_rope_params & from, const llama_rope_params & to) {
    int sections[4];
    memcpy(sections, tc.sections, sizeof(sections));

    return llama_rope_level_move(ctx, x, pos, pos_neg, nullptr, tc.n_rot, tc.mrope ? sections : nullptr, tc.mode,
            n_ctx_orig, freq_base, from, to, 32.0f, 1.0f);
}

// build with fn on a fresh context, allocate on the backend, compute
template <typename F>
static void compute(ggml_backend_t backend, F && fn) {
    ggml_init_params ip = { ggml_tensor_overhead()*256 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    fn(ctx, gf);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
}

static std::vector<float> to_f32(const ggml_tensor * t) {
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

    std::vector<float> res(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(res.data(), raw.data(), raw.size());
    } else {
        ggml_get_type_traits(t->type)->to_float(raw.data(), res.data(), res.size());
    }

    return res;
}

// returns the number of failures
static int run(ggml_backend_t backend, const test_case & tc, ggml_type type_cache, float tol) {
    const int n_head  = 2;
    const int n_cells = 96;
    const int n_move  = 40; // cells [8, 48) move, the others belong to other sequences and must stay
    const int n_pos   = tc.mrope ? 4 : 1;
    const int64_t row = (int64_t) tc.n_embd_head*n_head;

    // the persistent tensors
    ggml_init_params ip = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * k       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.n_embd_head, n_head, n_cells);
    ggml_tensor * pos     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cells*n_pos);
    ggml_tensor * pos_neg = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cells*n_pos);
    ggml_tensor * idxs    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_move);
    ggml_tensor * mpos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_move*n_pos);
    ggml_tensor * mpos_ng = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_move*n_pos);
    ggml_tensor * cache   = ggml_new_tensor_2d(ctx, type_cache, row, n_cells); // as llama_kv_cache holds it
    ggml_tensor * ref     = ggml_new_tensor_2d(ctx, type_cache, row, n_cells); // the new level, stored the same way
    ggml_tensor * rt      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.n_embd_head, n_head, n_cells);
    ggml_tensor * rt_ref  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.n_embd_head, n_head, n_cells);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<float> k_h(ggml_nelements(k));
    for (auto & v : k_h) {
        v = ud(rng);
    }
    ggml_backend_tensor_set(k, k_h.data(), 0, ggml_nbytes(k));

    // positions far past the native context, a few image-like cells with their own y, x and 4th position
    std::vector<int32_t> p(n_cells*n_pos), pn(n_cells*n_pos);
    for (int i = 0; i < n_cells; ++i) {
        const int32_t t = 261000 + 9973*i;
        for (int j = 0; j < n_pos; ++j) {
            p[j*n_cells + i] = j < 3 ? t : 0;
        }
        if (tc.mrope && i % 7 == 3) {
            p[1*n_cells + i] = t + i % 5;
            p[2*n_cells + i] = t + i % 11;
            p[3*n_cells + i] = i;
        }
    }
    for (size_t i = 0; i < p.size(); ++i) {
        pn[i] = -p[i];
    }
    ggml_backend_tensor_set(pos,     p.data(),  0, ggml_nbytes(pos));
    ggml_backend_tensor_set(pos_neg, pn.data(), 0, ggml_nbytes(pos_neg));

    std::vector<int32_t> ih(n_move), mp(n_move*n_pos), mn(n_move*n_pos);
    for (int i = 0; i < n_move; ++i) {
        ih[i] = 8 + i;
        for (int j = 0; j < n_pos; ++j) {
            mp[j*n_move + i] =  p[j*n_cells + 8 + i];
            mn[j*n_move + i] = -p[j*n_cells + 8 + i];
        }
    }
    ggml_backend_tensor_set(idxs,    ih.data(), 0, ggml_nbytes(idxs));
    ggml_backend_tensor_set(mpos,    mp.data(), 0, ggml_nbytes(mpos));
    ggml_backend_tensor_set(mpos_ng, mn.data(), 0, ggml_nbytes(mpos_ng));

    int failures = 0;

    const llama_rope_params * pairs[][2] = {
        { &plain, &yarn  },
        { &yarn,  &plain },
        { &plain, &yarn2 },
        { &yarn2, &plain },
    };

    for (auto & pr : pairs) {
        const llama_rope_params & from = *pr[0];
        const llama_rope_params & to   = *pr[1];

        compute(backend, [&](ggml_context * c, ggml_cgraph * gf) {
            ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_2d(c, rope(c, k, pos, tc, from), row, n_cells), cache));
            ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_2d(c, rope(c, k, pos, tc, to),   row, n_cells), ref));
        });

        const std::vector<float> before = to_f32(cache);

        // the move graph of llama_kv_cache::build_graph_rope_level, on cells [8, 8 + n_move)
        compute(backend, [&](ggml_context * c, ggml_cgraph * gf) {
            ggml_tensor * cur = ggml_get_rows(c, cache, idxs);
            cur = ggml_reshape_3d(c, cur, tc.n_embd_head, n_head, n_move);
            cur = move(c, cur, mpos, mpos_ng, tc, from, to);
            cur = ggml_reshape_2d(c, cur, row, n_move);
            ggml_build_forward_expand(gf, ggml_set_rows(c, cache, cur, idxs));
        });

        const std::vector<float> got  = to_f32(cache);
        const std::vector<float> want = to_f32(ref);

        double emax = 0.0, amax = 0.0;
        bool untouched = true;
        for (int i = 0; i < n_cells; ++i) {
            const bool moved = i >= 8 && i < 8 + n_move;
            for (int64_t j = 0; j < row; ++j) {
                const float g = got[i*row + j];
                if (moved) {
                    emax = std::max(emax, (double) fabsf(g - want[i*row + j]));
                    amax = std::max(amax, (double) fabsf(want[i*row + j]));
                } else if (g != before[i*row + j]) {
                    untouched = false;
                }
            }
        }

        const float err = (float) (emax/amax);
        const bool  ok  = err <= tol && untouched;

        printf("  %-22s %-4s %s -> %s (attn %.2f -> %.2f): max rel err %.3g, other cells %s: %s\n", tc.name, ggml_type_name(type_cache),
                from.ext_factor != 0.0f ? "yarn " : "plain", to.ext_factor != 0.0f ? "yarn " : "plain",
                from.attn_factor, to.attn_factor, err, untouched ? "unchanged" : "CHANGED", ok ? "OK" : "FAIL");

        failures += ok ? 0 : 1;
    }

    // a move there and back is the identity
    if (type_cache == GGML_TYPE_F32) {
        compute(backend, [&](ggml_context * c, ggml_cgraph * gf) {
            ggml_tensor * a = rope(c, k, pos, tc, plain);
            ggml_build_forward_expand(gf, ggml_cpy(c, a, rt_ref));
            ggml_build_forward_expand(gf, ggml_cpy(c, move(c, move(c, a, pos, pos_neg, tc, plain, yarn2), pos, pos_neg, tc, yarn2, plain), rt));
        });

        const std::vector<float> a = to_f32(rt_ref);
        const std::vector<float> b = to_f32(rt);

        double emax = 0.0, amax = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            emax = std::max(emax, (double) fabsf(a[i] - b[i]));
            amax = std::max(amax, (double) fabsf(a[i]));
        }

        const float err = (float) (emax/amax);
        const bool  ok  = err <= 1e-5f;
        printf("  %-22s f32  round trip plain -> yarn -> plain: max rel err %.3g: %s\n", tc.name, err, ok ? "OK" : "FAIL");
        failures += ok ? 0 : 1;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return failures;
}

int main(void) {
    ggml_backend_load_all();

    const test_case cases[] = {
        { "imrope n_rot 64/256",  GGML_ROPE_TYPE_IMROPE, 256,  64, { 11, 11, 10, 0 }, true  },
        { "imrope n_rot 128/128", GGML_ROPE_TYPE_IMROPE, 128, 128, { 24, 20, 20, 0 }, true  },
        { "mrope n_rot 128/128",  GGML_ROPE_TYPE_MROPE,  128, 128, { 16, 24, 24, 0 }, true  },
        { "neox n_rot 64/128",    GGML_ROPE_TYPE_NEOX,   128,  64, {  0,  0,  0, 0 }, false },
        { "norm n_rot 128/128",   0,                     128, 128, {  0,  0,  0, 0 }, false },
    };

    int failures = 0;

    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend);

        printf("%s:\n", ggml_backend_dev_name(dev));

        for (const auto & tc : cases) {
            failures += run(backend, tc, GGML_TYPE_F32,  1e-5f);
            failures += run(backend, tc, GGML_TYPE_F16,  2e-3f);
            failures += run(backend, tc, GGML_TYPE_Q8_0, 2e-2f); // re-quantized once more than the reference
        }

        ggml_backend_free(backend);
    }

    if (failures) {
        printf("%d FAILED\n", failures);
        return 1;
    }

    printf("all OK\n");
    return 0;
}
