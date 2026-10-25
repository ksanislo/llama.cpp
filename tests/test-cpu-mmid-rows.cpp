// MUL_MAT_ID with a few rows per expert on the CPU backend: the default path (on AVX1 CPUs the
// multi-row dots of ggml-cpu/arch/x86/quants.c) must match the reference per-row vec_dot path
// (use_ref) bit for bit

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint64_t rng_next() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static float rng_f32() {
    return (float) (rng_next() % 20001) * 1e-4f - 1.0f;
}

static void cpu_set_use_ref(ggml_backend_t backend, bool use_ref) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_use_ref)(ggml_backend_t, bool) =
        (void (*)(ggml_backend_t, bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
    if (!set_use_ref) {
        fprintf(stderr, "ggml_backend_cpu_set_use_ref not available\n");
        exit(1);
    }
    set_use_ref(backend, use_ref);
}

static void cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_n_threads)(ggml_backend_t, int) =
        (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (!set_n_threads) {
        fprintf(stderr, "ggml_backend_set_n_threads not available\n");
        exit(1);
    }
    set_n_threads(backend, n_threads);
}

// rows_per_expert[e] tokens are routed to expert e (n_used slots per token, the slot each lands in rotates)
static bool test_case(ggml_backend_t backend, ggml_type type, int64_t K, int64_t N, int n_used,
                      const std::vector<int> & rows_per_expert) {
    const int n_expert = (int) rows_per_expert.size();

    // assignments (expert per slot): fill tokens slot by slot so no token picks one expert twice
    std::vector<int> assign;
    for (int e = 0; e < n_expert; ++e) {
        for (int k = 0; k < rows_per_expert[e]; ++k) {
            assign.push_back(e);
        }
    }
    const int64_t n_tokens = ((int64_t) assign.size() + n_used - 1) / n_used;
    std::vector<int32_t> ids(n_tokens * n_used, -1);
    {
        // column-major fill: token t, slot u gets assign[u*n_tokens + t] -> one expert never repeats within a token
        // as long as every expert has at most n_tokens rows
        for (size_t a = 0; a < assign.size(); ++a) {
            const int64_t u = (int64_t) a / n_tokens;
            const int64_t t = (int64_t) a % n_tokens;
            ids[t*n_used + u] = assign[a];
        }
        // pad: unused slots get an expert this token does not use yet (an extra row for it)
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int u = 0; u < n_used; ++u) {
                if (ids[t*n_used + u] >= 0) {
                    continue;
                }
                for (int e = 0; e < n_expert; ++e) {
                    bool used = false;
                    for (int v = 0; v < n_used; ++v) {
                        used = used || ids[t*n_used + v] == e;
                    }
                    if (!used) {
                        ids[t*n_used + u] = e;
                        break;
                    }
                }
            }
        }
    }

    struct ggml_init_params ip = { 256*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * as  = ggml_new_tensor_3d(ctx, type,           K, N, n_expert);
    struct ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  K, n_used, n_tokens);
    struct ggml_tensor * idt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,  n_used, n_tokens);
    struct ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, idt);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    {
        const int64_t rows = N*n_expert;
        std::vector<float> w(rows*K);
        for (auto & v : w) {
            v = rng_f32();
        }
        std::vector<float> im;
        if (ggml_quantize_requires_imatrix(type)) {
            im.assign(K, 1.0f);
        }
        std::vector<uint8_t> q(ggml_nbytes(as));
        ggml_quantize_chunk(type, w.data(), q.data(), 0, rows, K, im.empty() ? nullptr : im.data());
        ggml_backend_tensor_set(as, q.data(), 0, q.size());

        std::vector<float> x(ggml_nelements(b));
        for (auto & v : x) {
            v = rng_f32();
        }
        ggml_backend_tensor_set(b, x.data(), 0, ggml_nbytes(b));
        ggml_backend_tensor_set(idt, ids.data(), 0, ggml_nbytes(idt));
    }

    std::vector<float> ref(ggml_nelements(out)), got(ggml_nelements(out));
    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(out, ref.data(), 0, ggml_nbytes(out));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

    int64_t n_diff = 0, first = -1;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            if (first < 0) {
                first = (int64_t) i;
            }
            ++n_diff;
        }
    }

    printf("%-8s K=%5lld N=%4lld n_used=%d rows/expert={", ggml_type_name(type), (long long) K, (long long) N, n_used);
    for (int e = 0; e < n_expert; ++e) {
        printf("%s%d", e ? "," : "", rows_per_expert[e]);
    }
    printf("}: %s", n_diff == 0 ? "OK (bit-identical)" : "FAIL");
    if (n_diff) {
        printf(" (%lld of %zu differ, first at %lld: ref %.9g got %.9g)", (long long) n_diff, ref.size(), (long long) first,
               ref[first], got[first]);
    }
    printf("\n");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return n_diff == 0;
}

int main(void) {
    // the tiled path is not what is tested here (it takes >= 8 rows when enabled)
#if defined(_WIN32)
    _putenv("GGML_CPU_TILED_MM=0");
#else
    unsetenv("GGML_CPU_TILED_MM");
    unsetenv("GGML_CPU_TILED_MM_FORCE");
#endif

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        fprintf(stderr, "failed to initialize the CPU backend\n");
        return 1;
    }

    const ggml_type types[] = {
        GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
    };

    int n_fail = 0;
    for (int n_threads : { 1, 4 }) {
        cpu_set_n_threads(backend, n_threads);
        printf("n_threads = %d\n", n_threads);
        for (ggml_type type : types) {
            // 1..7 rows per expert, src0 rows not a multiple of the 16-row block
            n_fail += !test_case(backend, type, 512, 70, 1, { 1, 2, 3, 4, 5, 6, 7 });
            n_fail += !test_case(backend, type, 768, 33, 2, { 2, 3, 7, 5, 4, 6, 1, 2 });
        }
        // Q8_0 / Q5_1 have no tiled path: their multi-row dots also run at >= 8 rows; an odd count of
        // 32-value blocks covers the q8_0 tail
        n_fail += !test_case(backend, GGML_TYPE_Q8_0, 544, 40, 1, { 9, 13, 3 });
        n_fail += !test_case(backend, GGML_TYPE_Q5_1, 544, 40, 2, { 9, 13, 21, 3 });
    }

    ggml_backend_free(backend);

    printf("%s\n", n_fail == 0 ? "all bit-identical" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
