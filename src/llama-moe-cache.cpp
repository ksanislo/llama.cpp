#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;
    int32_t n_filled    = 0; // slots that hold an expert

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    // take slot s out of the lists, it is no longer a victim
    void unlink(int32_t s) {
        const bool from = is_protected[s];
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }
        prev[s] = next[s] = -1;
        is_protected[s] = false;
        n_protected -= int(from);
    }

    // put the unlinked slot d at the place of slot s in the lists, s is unlinked
    void replace(int32_t s, int32_t d) {
        const bool seg = is_protected[s];
        prev[d] = prev[s];
        next[d] = next[s];
        if (prev[s] >= 0) {
            next[prev[s]] = d;
        } else {
            head[seg] = d;
        }
        if (next[s] >= 0) {
            prev[next[s]] = d;
        } else {
            tail[seg] = d;
        }
        is_protected[d] = seg;
        prev[s] = next[s] = -1;
        is_protected[s] = false;
    }

    // put the unlinked slot s at the least recently used end of probation, it is the next victim
    void push_front(int32_t s) {
        prev[s] = -1;
        next[s] = head[0];
        if (head[0] >= 0) {
            prev[head[0]] = s;
        } else {
            tail[0] = s;
        }
        head[0] = s;
        is_protected[s] = false;
    }

    // put the unlinked slot s at the most recently used end of probation
    void push_back(int32_t s) {
        prev[s] = tail[0];
        next[s] = -1;
        if (tail[0] >= 0) {
            next[tail[0]] = s;
        } else {
            head[0] = s;
        }
        tail[0] = s;
        is_protected[s] = false;
    }

    // mark the distinct experts of ids as selected in layer il (see victim), returns them in uniq
    void mark_seen(const int32_t * ids, size_t n_ids) {
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t try_victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        const int32_t s = try_victim(il);
        if (s < 0) {
            GGML_ABORT("no SLRU victim");
        }
        return s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int32_t n_used, int32_t n_zero, F && n_cpu, uint64_t & rng,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        mark_seen(ids, n_ids);
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            if (slots[e] < n_slots) {
                touch(slots[e]);
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the misses left to the CPU are picked at random, so that every expert can enter the cache
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        for (size_t i = 0; i < n_c; ++i) {
            rng = rng*6364136223846793005ull + 1442695040888963407ull;
            std::swap(misses[i], misses[i + (rng >> 33) % (misses.size() - i)]);
        }
        if (n_c > 0) {
            const size_t n_tokens = n_ids/n_used;
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (size_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                const int32_t z = used == ~0ull ? 64 : __builtin_ctzll(~used);
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (size_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = empty;
            } else {
                n_filled++;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }

    // give a slot to each predicted expert e of layer il_next that is not cached, at most max_fill, appended to fills
    // the victims spare the experts selected by the last plan if it was for layer il (its device part may still read them)
    // a prefetched expert enters probation like a fill, so a wrong prediction is the next victim
    void prefetch(int32_t il, int32_t il_next, const int32_t * pred, size_t n_pred, size_t max_fill, std::vector<fill> & fills) {
        int32_t * slots = slot_map[il_next];
        for (size_t i = 0; i < n_pred && fills.size() < max_fill; ++i) {
            const int32_t e = pred[i];
            if (e < 0 || e >= n_expert || slots[e] < n_slots) {
                continue;
            }
            // a cached guess is not a victim: it goes to the recent end of probation, like a new one
            int32_t s = victim(il);
            for (size_t n_moved = 0; key_of[s] >= 0 && key_of[s] / n_expert == il_next && n_moved < n_pred &&
                    std::find(pred, pred + n_pred, key_of[s] % n_expert) != pred + n_pred; ++n_moved) {
                move(s, false);
                s = victim(il);
            }
            if (std::any_of(fills.begin(), fills.end(), [&](const fill & f) { return f.slot == s; }) ||
                (key_of[s] >= 0 && key_of[s] / n_expert == il_next && std::find(pred, pred + n_pred, key_of[s] % n_expert) != pred + n_pred)) {
                break; // the cache is too small to hold more guesses
            }
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = empty;
            } else {
                n_filled++;
            }
            key_of[s] = il_next*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

}

static ggml_context_ptr llama_moe_cache_init_ctx(size_t n_tensors) {
    ggml_init_params params = {
        /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr res(ggml_init(params));
    if (!res) {
        throw std::runtime_error("failed to create the MoE cache context");
    }
    return res;
}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // staging (LLAMA_MOE_ASYNC): the cached layers of the device in graph order, layer layers[i] is staged in slot i % n_stage
        std::vector<int32_t> layers;
        ggml_backend_ptr copy;                        // a second backend of the device, its stream uploads to the slots
        ggml_backend_event_ptr ev_compute;            // position of the compute stream when the copy stream last started on it
        std::vector<ggml_backend_event_ptr> ev_ready; // [n_stage] the last upload to the slot is done
        std::vector<int32_t> slot_layer;              // [n_stage] the layer whose experts are all in the slot, -1 if none
        int32_t last_slot = -1;                       // the slot of the last upload, -1 if the compute stream already waited for it
        size_t slot_size = 0;
        ggml_context_ptr ctx_stage;
        ggml_backend_buffer_ptr buf_stage;            // nullptr if the slots borrow the cache

        // LLAMA_MOE_ASYNC_BORROW: the slots lie in the bank tails of the lender group, from its slot borrow_first on
        int32_t lender = -1;
        int32_t borrow_first = 0;
        bool borrowed = false;                        // the cache slots under the staging slots are out of the SLRU

        // CPU part results while an earlier graph may still run: copied to the device from here, not from the CPU compute buffer
        ggml_backend_buffer_ptr buf_out;
        bool out_idle   = true;  // no copy from buf_out is queued (the device was synchronized after the last one)
        bool out_direct = false; // a copy from the CPU compute buffer may be queued

        // LLAMA_MOE_PREFETCH: uploads of predicted experts on the copy stream, one expert after the other, best guess first
        ggml_backend_event_ptr ev_fills;                // the compute stream after the fills of the current layer, the prefetch starts after it
        std::vector<ggml_backend_event_ptr> ev_pf;      // [pref_max] the copy stream after upload i of the last prefetch
        ggml_backend_event_ptr ev_pf_prev;              // the copy stream after the uploads of the earlier prefetches
        std::vector<int32_t> pf_slot;                   // slots of the last prefetch, in upload order
        int32_t pf_layer = -1;                          // layer of the last prefetch
        bool pf_pending      = false;                   // the compute stream may not have waited for all uploads of the last prefetch
        bool pf_prev_pending = false;                   // ... or for ev_pf_prev
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_slru slru;
        std::vector<std::vector<ggml_tensor *>> stage; // [n_stage][expert tensor] staging slots, n_expert + 1 experts each
        std::vector<bool> pf_unused;      // [n_slots] prefetched and not selected since (LLAMA_MOE_PREFETCH statistics)
        std::vector<bool> dh_ok;                       // [dh_max + 1] the device supports the device-side path for n tokens
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks

        int32_t rank = -1;                  // index in the layers of its device
        ggml_tensor * stage_map = nullptr;  // I32 [1, n_expert] in host memory, the identity: a staged expert is read at its own index

        next_router router;                 // LLAMA_MOE_PREFETCH: the router of the next layer, if it is cached on the same device
        // LLAMA_MOE_DEVICE_HITS
        int32_t dh_row = -1;                 // row in the device map and log of its device, -1 if the layer has no device path
        ggml_tensor * dh_map = nullptr;      // I32 [1, n_expert] on the device: the slot of a cached expert, -1 - e for the others
        ggml_tensor * dh_log = nullptr;      // I32 [dh_log_cap] on the device: the experts selected by the last graph, -1 after them
        std::vector<ggml_tensor *> dh_alias; // the host experts as tensors of the device (mapped pinned memory), in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t bytes  = 0;
    };

    struct stage_stats {
        size_t layers_prefetched = 0; // staged before the layer was reached
        size_t layers_waited     = 0; // staged when the layer was reached
        size_t layers_partial    = 0; // only the selected experts staged, after reading them back
        size_t prefetches        = 0;
        size_t bytes_host        = 0;
        size_t bytes_cache       = 0;
        size_t borrows           = 0;
        size_t experts_moved     = 0; // kept by moving them out of the borrowed slots
        size_t experts_evicted   = 0;
    };

    // device data read by the CPU part of a layer
    struct stash_entry {
        const ggml_tensor * base; // the tensor that owns the data
        size_t offs;              // offset of the data in base
        size_t size;
        size_t pos;               // position in stash_buf
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    int32_t n_stage  = 0; // staging slots per device, 0 if the batches that the cache does not serve use op offload
    int64_t full_ids = 8; // ids per expert from which a batch stages all the experts of a layer without reading the ids back
    bool    zero_rows = true; // LLAMA_MOE_CACHE_ZERO_ROWS=0: a lookup of 0 tokens bypasses the cache (graphs reallocate, see use_cache)
    bool    borrow   = false; // the staging slots take cache slots during large batches instead of their own memory
    stage_stats stats_stage;
    std::unordered_map<const ggml_tensor *, int32_t> stage_layer_of;              // staging map -> layer
    std::unordered_map<const ggml_tensor *, ggml_tensor *> stage_bindings;       // host experts -> their staging slot
    std::vector<uint8_t> zeros = std::vector<uint8_t>(512, 0);                    // padding after the last expert of a slot

    // LLAMA_MOE_PREFETCH
    int32_t n_pref   = 0; // predicted experts per token, 0 if off
    int32_t pref_max = 0; // uploaded experts per layer at most
    struct prediction {
        const ggml_tensor * pred; // I32 [n_pref, n_tokens] on the device
    };
    std::unordered_map<const ggml_tensor *, prediction> predictions; // slot lookup -> predicted experts of the next layer
    std::vector<int32_t> pred_ids;
    std::vector<int32_t> ids_pf;
    std::vector<moe_cache_slru::fill> pf_fills;
    struct pf_stats {
        size_t layers = 0; // layers with a prediction read back
        size_t issued = 0; // experts uploaded ahead
        size_t used   = 0; // of them, selected by the next lookup of their layer
        size_t wasted = 0; // evicted before being selected
        size_t bytes  = 0;
    } stats_pf;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<bool>    used;
    std::vector<moe_cache_slru::fill> fills;
    std::vector<moe_cache_slru::fill> cpu;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group
    uint64_t rng       = 0x853c49e6748fea9bull;

    // the CPU part of the layer of a slot lookup
    struct cpu_part {
        int32_t n_used; // experts per token
        std::vector<const ggml_tensor *> inputs; // device tensors read by the CPU part
    };
    std::unordered_map<const ggml_tensor *, cpu_part> cpu_parts; // slot lookup -> CPU part, no inputs if there is no CPU part
    std::unordered_set<const ggml_tensor *> cpu_outputs; // results of the CPU parts, copied to the device asynchronously

    // the device data read with the selected experts of the last slot lookup, in host memory that the device can copy to asynchronously
    std::vector<stash_entry> stash;
    ggml_backend_buffer_ptr stash_buf;

    // the graph may start before the previous one is done (pipeline parallelism, or ubatches without a synchronization)
    bool out_buffered = false;

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // LLAMA_MOE_DEVICE_HITS=1|2: batches of up to dh_max tokens look up the slots of the selected experts on the device (a device
    // copy of the slot maps) and read the cached experts from the banks and the others straight from the pinned host experts over
    // PCIe, in the same kernel: no read back of the selected experts and no host step in the graph, so a decode token is one graph
    // per device. The graph also logs the selected experts on the device; the next graph's begin() reads the log (one
    // synchronization per device per graph, where the decode synchronizes anyway), updates the SLRU and uploads the missed experts
    // on a separate stream: 1 = after the device's part of that graph (no PCIe contention with its own reads), 2 = right away.
    // A fill is used from the graph after its upload is done; until then the expert is read from host memory.
    struct dh_fill {
        int32_t il;
        int32_t expert;
        int32_t slot;
    };
    struct dh_device {
        ggml_context_ptr ctx;
        ggml_context_ptr ctx_alias;
        ggml_backend_buffer_ptr buf;
        std::vector<ggml_backend_buffer_ptr> mapped; // the host expert buffers mapped to the device
        ggml_tensor * map = nullptr;                 // I32 [n_expert, n_rows]
        ggml_tensor * log = nullptr;                 // I32 [dh_log_cap, n_rows]
        std::vector<int32_t> rows;                   // [n_rows] layer of each row
        std::vector<int32_t> map_host;
        std::vector<int32_t> log_host;
        std::vector<int32_t> log_clear;              // all -1
        bool map_dirty = false;
        bool resync    = true;                       // the host slot maps were changed by the cache path, rebuild the device map
        ggml_backend_ptr fill;                       // a second backend of the device, its stream uploads the fills
        ggml_backend_event_ptr ev_after;             // compute stream position the fills wait for
        ggml_backend_event_ptr ev_fill;              // the fills are done
        std::vector<dh_fill> pending;                // slots taken, not uploaded yet
        std::vector<dh_fill> queued;                 // uploading, committed at the next begin()
    };
    struct dh_stats {
        size_t graphs  = 0;
        size_t hits    = 0;
        size_t misses  = 0; // read from host memory
        size_t fills   = 0;
        size_t dropped = 0; // fills made redundant by the cache path, which uploaded the expert itself
        size_t bytes_miss = 0;
        size_t bytes_fill = 0;
    };
    int32_t dh_mode     = 0;
    int64_t dh_max      = 8;  // tokens per ubatch
    int32_t dh_fill_max = 64; // fills per device per graph
    int64_t dh_log_cap  = 0;
    bool    dh_logged   = false; // the last graph logs selected experts
    std::vector<dh_device> dh_devs; // [devices]
    dh_stats stats_dh;
    std::unordered_set<const ggml_tensor *> dh_logs;                  // the logs, to find the graphs that write them
    std::unordered_set<int32_t> dh_inflight;                          // il*n_expert + e of the pending and queued fills
    std::vector<int32_t> dh_cand;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> dh_alias_of; // host experts -> mapped

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        if (const char * env = getenv("LLAMA_MOE_ASYNC")) {
            n_stage = std::max(0, atoi(env));
        }
        if (const char * env = getenv("LLAMA_MOE_ASYNC_FULL")) {
            full_ids = std::max<int64_t>(1, atoll(env));
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_ZERO_ROWS")) {
            zero_rows = atoi(env) != 0;
        }
        if (const char * env = getenv("LLAMA_MOE_ASYNC_BORROW")) {
            borrow = atoi(env) != 0;
        }
        if (const char * env = getenv("LLAMA_MOE_PREFETCH")) {
            n_pref = std::clamp(atoi(env), 0, (int) model.hparams.n_expert);
            pref_max = n_pref;
            if (const char * env_max = getenv("LLAMA_MOE_PREFETCH_MAX")) {
                pref_max = std::max(0, atoi(env_max));
            }
            if (pref_max == 0) {
                n_pref = 0;
            }
        }
        for (const device & d : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(ggml_backend_get_device(d.backend), &props);
            if (n_stage > 0 && (!props.caps.async || !props.caps.events)) {
                LLAMA_LOG_WARN("%s: %s does not support async copies and events, LLAMA_MOE_ASYNC is disabled\n", __func__, ggml_backend_name(d.backend));
                n_stage = 0;
            }
            if (n_pref > 0 && (!props.caps.async || !props.caps.events)) {
                LLAMA_LOG_WARN("%s: %s does not support async copies and events, LLAMA_MOE_PREFETCH is disabled\n", __func__, ggml_backend_name(d.backend));
                n_pref = 0;
            }
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_tensor_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(1 + (n_stage > 0 ? 1 : 0) + (n_zero > 0 ? 1 : 0)); // slot map, staging map, CPU map
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = llama_moe_cache_init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = llama_moe_cache_init_ctx(n_tensors_host);
        ctx_views = llama_moe_cache_init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_stage > 0) {
                    l.stage_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.stage_map, "moe_cache.stage_map-%d", il);
                    stage_layer_of[l.stage_map] = il;
                    buf_host_size += GGML_PAD(ggml_nbytes(l.stage_map), alignment_host);
                }
                if (n_zero > 0) {
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.slru.empty);
                        g.slru.slot_map[il] = slots;
                    }
                    if (layers[il].stage_map != nullptr) {
                        int32_t * map = (int32_t *) layers[il].stage_map->data;
                        for (int64_t e = 0; e < layers[il].stage_map->ne[1]; ++e) {
                            map[e] = (int32_t) e;
                        }
                    }
                }
            }
        }

        if (n_stage > 0) {
            init_stage(model.hparams.no_alloc);
        }
        if (n_pref > 0 && dh_requested()) {
            // device hits have no per-layer host step for the prediction readback and fill the cache from their own log
            LLAMA_LOG_WARN("%s: LLAMA_MOE_PREFETCH is disabled, LLAMA_MOE_DEVICE_HITS serves the decode batches\n", __func__);
            n_pref = 0;
        }
        if (n_pref > 0) {
            init_prefetch(model);
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            if (d.slot_size > 0) {
                LLAMA_LOG_INFO("%s: %10s MoE staging   = %8.2f MiB, %d slots for %zu layers (LLAMA_MOE_ASYNC, all experts from %" PRId64 " ids per expert)\n", __func__,
                    ggml_backend_buft_name(d.buft), n_stage*d.slot_size/1024.0/1024.0, n_stage, d.layers.size(), full_ids);
            }
            if (d.lender >= 0) {
                const group & g = groups[d.lender];
                LLAMA_LOG_INFO("%s: %10s MoE staging borrows %d of the %d %s slots during large batches (LLAMA_MOE_ASYNC_BORROW)\n", __func__,
                    ggml_backend_buft_name(d.buft), g.n_slots - d.borrow_first, g.n_slots, ggml_type_name(g.ref.back()->type));
            }
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }

        if (const char * env = getenv("LLAMA_MOE_DEVICE_HITS")) {
            dh_mode = std::clamp(atoi(env), 0, 2);
        }
        if (const char * env = getenv("LLAMA_MOE_DEVICE_HITS_MAX")) {
            dh_max = std::clamp<int64_t>(atoll(env), 1, max_batch);
        }
        if (const char * env = getenv("LLAMA_MOE_DEVICE_HITS_FILLS")) {
            dh_fill_max = std::max(0, atoi(env));
        }
        if (dh_mode > 0 && !model.hparams.no_alloc) {
            init_device_hits();
        }
    }

    // LLAMA_MOE_DEVICE_HITS is set and not ruled out by the cache's settings (see init_device_hits)
    bool dh_requested() const {
        const char * env = getenv("LLAMA_MOE_DEVICE_HITS");
        return env != nullptr && atoi(env) > 0 && cpu_share <= 0.0f && !borrow;
    }

    void init_device_hits() {
        if (cpu_share > 0.0f || borrow) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_DEVICE_HITS does not combine with --moe-cache-cpu-share or LLAMA_MOE_ASYNC_BORROW, it is disabled\n", __func__);
            dh_mode = 0;
            return;
        }
        dh_log_cap = dh_max*n_expert_used;
        dh_devs.resize(devices.size());

        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            dh_device & dd = dh_devs[id];
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);

            typedef ggml_backend_buffer_t (*from_mapped_t)(ggml_backend_dev_t, void *, size_t);
            auto from_mapped = (from_mapped_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_buffer_from_mapped_host_ptr");

            int64_t n_expert_max = 0;
            size_t n_alias = 0;
            for (size_t il = 0; il < layers.size(); ++il) {
                const layer & l = layers[il];
                if (l.ig >= 0 && groups[l.ig].id == (int32_t) id && groups[l.ig].n_slots > 0) {
                    dd.rows.push_back((int32_t) il);
                    n_expert_max = std::max(n_expert_max, l.slot_map->ne[1]);
                    n_alias += l.experts.size();
                }
            }
            if (dd.rows.empty()) {
                continue;
            }
            if (from_mapped == nullptr) {
                LLAMA_LOG_WARN("%s: %s cannot read host memory directly, LLAMA_MOE_DEVICE_HITS is off there\n", __func__, ggml_backend_name(d.backend));
                dd.rows.clear();
                continue;
            }

            // the host experts as device tensors: one mapped buffer per host buffer
            dd.ctx_alias = llama_moe_cache_init_ctx(n_alias);
            std::unordered_map<ggml_backend_buffer_t, ggml_backend_buffer_t> mapped_of;
            std::vector<int32_t> rows;
            for (int32_t il : dd.rows) {
                layer & l = layers[il];
                bool ok = true;
                for (ggml_tensor * w : l.experts) {
                    ggml_backend_buffer_t hb = w->buffer;
                    auto it = mapped_of.find(hb);
                    if (it == mapped_of.end()) {
                        ggml_backend_buffer_t mb = from_mapped(dev, ggml_backend_buffer_get_base(hb), ggml_backend_buffer_get_size(hb));
                        if (mb != nullptr) {
                            ggml_backend_buffer_set_usage(mb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                            dd.mapped.emplace_back(mb);
                        }
                        it = mapped_of.emplace(hb, mb).first;
                    }
                    if (it->second == nullptr || !ggml_is_contiguous(w)) {
                        ok = false;
                        break;
                    }
                    ggml_tensor * a = ggml_new_tensor_3d(dd.ctx_alias.get(), w->type, w->ne[0], w->ne[1], w->ne[2]);
                    GGML_ASSERT(a->nb[1] == w->nb[1] && a->nb[2] == w->nb[2]);
                    ggml_format_name(a, "moe_mapped.%s", w->name);
                    // placed by hand: the device buffer type would pad quantized rows past the host tensor
                    const size_t offs = (const uint8_t *) w->data - (const uint8_t *) ggml_backend_buffer_get_base(hb);
                    if (offs + ggml_nbytes(w) > ggml_backend_buffer_get_size(it->second)) {
                        ok = false;
                        break;
                    }
                    a->buffer = it->second;
                    a->data   = (uint8_t *) ggml_backend_buffer_get_base(it->second) + offs;
                    l.dh_alias.push_back(a);
                }
                if (!ok) {
                    l.dh_alias.clear();
                    LLAMA_LOG_WARN("%s: the experts of layer %d are not in pinned host memory %s can read (use --load-mode none), "
                        "they keep the cache path\n", __func__, il, ggml_backend_name(d.backend));
                    continue;
                }
                rows.push_back(il);
            }
            dd.rows = rows;
            if (dd.rows.empty()) {
                continue;
            }

            // the device map and the log, one row per layer
            const int64_t n_rows = (int64_t) dd.rows.size();
            dd.ctx = llama_moe_cache_init_ctx(2 + 2*n_rows);
            dd.map = ggml_new_tensor_2d(dd.ctx.get(), GGML_TYPE_I32, n_expert_max, n_rows);
            dd.log = ggml_new_tensor_2d(dd.ctx.get(), GGML_TYPE_I32, dh_log_cap, n_rows);
            ggml_format_name(dd.map, "moe_cache.device_map.%zu", id);
            ggml_format_name(dd.log, "moe_cache.device_log.%zu", id);
            for (int64_t r = 0; r < n_rows; ++r) {
                layer & l = layers[dd.rows[r]];
                l.dh_row = (int32_t) r;
                l.dh_map = ggml_view_2d(dd.ctx.get(), dd.map, 1, l.slot_map->ne[1], sizeof(int32_t), r*dd.map->nb[1]);
                l.dh_log = ggml_view_1d(dd.ctx.get(), dd.log, dh_log_cap, r*dd.log->nb[1]);
                ggml_format_name(l.dh_map, "moe_cache.device_map-%d", dd.rows[r]);
                ggml_format_name(l.dh_log, "moe_cache.device_log-%d", dd.rows[r]);
            }
            dd.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(dd.ctx.get(), d.buft));
            if (!dd.buf) {
                throw std::runtime_error("failed to allocate the MoE cache device maps");
            }
            dh_logs.insert(dd.log);
            dd.map_host.assign(ggml_nelements(dd.map), -1);
            dd.log_host.assign(ggml_nelements(dd.log), -1);
            dd.log_clear.assign(ggml_nelements(dd.log), -1);
            dh_rebuild_map(dd);
            ggml_backend_tensor_set(dd.map, dd.map_host.data(), 0, ggml_nbytes(dd.map));
            ggml_backend_tensor_set(dd.log, dd.log_clear.data(), 0, ggml_nbytes(dd.log));
            dd.map_dirty = false;
            dd.resync    = false;

            dd.fill.reset(ggml_backend_dev_init(dev, nullptr));
            dd.ev_after.reset(ggml_backend_event_new(dev));
            dd.ev_fill.reset(ggml_backend_event_new(dev));
            if (!dd.fill || !dd.ev_after || !dd.ev_fill) {
                throw std::runtime_error("failed to create the MoE cache fill stream");
            }
            for (int32_t il : dd.rows) {
                for (size_t ip = 0; ip < layers[il].experts.size(); ++ip) {
                    dh_alias_of[layers[il].experts[ip]] = layers[il].dh_alias[ip];
                }
            }
        }

        // which batch sizes the device supports, per group (the same expert types for all its layers)
        for (group & g : groups) {
            g.dh_ok.assign(dh_max + 1, false);
            if (g.n_slots == 0) {
                continue;
            }
            const layer & l0 = layers[g.layers[0]];
            if (l0.dh_row < 0) {
                continue;
            }
            const bool all_rows = std::all_of(g.layers.begin(), g.layers.end(), [&](int32_t il) { return layers[il].dh_row >= 0; });
            if (!all_rows) {
                continue; // the cache path of one layer may evict the slots of another, all or none use the device path
            }
            ggml_backend_t backend = devices[g.id].backend;
            for (int64_t n = 1; n <= dh_max; ++n) {
                ggml_context_ptr sctx = llama_moe_cache_init_ctx(32);
                ggml_context * ctx = sctx.get();
                bool ok = true;
                for (size_t ip = 0; ip < g.ref.size() && ok; ++ip) {
                    const ggml_tensor * w = g.ref[ip];
                    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], 1, n);
                    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n);
                    ggml_tensor * op  = ggml_mul_mat_id_mapped(ctx, bindings.at(l0.experts[ip]).cached, l0.dh_alias[ip], b, ids);
                    ok = ggml_backend_supports_op(backend, op);
                }
                ggml_tensor * ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n*n_expert_used);
                ok = ok && ggml_backend_supports_op(backend, ggml_get_rows(ctx, l0.dh_map, ids));
                ok = ok && ggml_backend_supports_op(backend, ggml_cpy(ctx, ids, ggml_view_1d(ctx, l0.dh_log, n*n_expert_used, 0)));
                g.dh_ok[n] = ok;
            }
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            const dh_device & dd = dh_devs[id];
            if (dd.rows.empty()) {
                continue;
            }
            int64_t n_ok = 0;
            for (int64_t n = dh_max; n >= 1 && n_ok == 0; --n) {
                bool all = true;
                for (int32_t il : dd.rows) {
                    all = all && groups[layers[il].ig].dh_ok[n];
                }
                n_ok = all ? n : 0;
            }
            LLAMA_LOG_INFO("%s: %10s MoE device hits: %zu layers, slots looked up on the device and misses read from host memory "
                "for up to %" PRId64 " tokens, fills %s (LLAMA_MOE_DEVICE_HITS=%d)\n", __func__, ggml_backend_name(devices[id].backend),
                dd.rows.size(), n_ok, dh_mode == 1 ? "after the device's part of the next graph" : "right away", dh_mode);
        }
    }

    // the device map from the host slot maps: the slot of a cached expert, -1 - e otherwise (also while its fill is in flight)
    void dh_rebuild_map(dh_device & dd) {
        for (size_t r = 0; r < dd.rows.size(); ++r) {
            const layer & l = layers[dd.rows[r]];
            const group & g = groups[l.ig];
            const int32_t * slots = g.slru.slot_map[dd.rows[r]];
            int32_t * row = dd.map_host.data() + r*dd.map->ne[0];
            for (int64_t e = 0; e < l.slot_map->ne[1]; ++e) {
                row[e] = slots[e] < g.n_slots ? slots[e] : -1 - (int32_t) e;
            }
        }
        dd.map_dirty = true;
    }

    void dh_map_set(int32_t il, int32_t e, int32_t v) {
        const layer & l = layers[il];
        if (l.dh_row < 0) {
            return;
        }
        dh_device & dd = dh_devs[groups[l.ig].id];
        dd.map_host[l.dh_row*dd.map->ne[0] + e] = v;
        dd.map_dirty = true;
    }

    // LLAMA_MOE_PREFETCH: a layer predicts the next one if both are cached on the same device, the router of the next one is on
    // that device too and its experts are pinned (an upload from pageable memory would block the host)
    void init_prefetch(const llama_model & model) {
        size_t n_pred_layers = 0;
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            ggml_backend_buffer_type_t buft_pinned = ggml_backend_dev_host_buffer_type(dev);
            bool any = false;
            for (size_t il = 0; il + 1 < layers.size(); ++il) {
                layer & l = layers[il];
                const layer & ln = layers[il + 1];
                if (l.ig < 0 || ln.ig < 0 || groups[l.ig].id != (int32_t) id || groups[ln.ig].id != (int32_t) id || groups[ln.ig].n_slots == 0) {
                    continue;
                }
                const llama_layer & mn = model.layers[il + 1];
                if (mn.ffn_gate_inp == nullptr || mn.ffn_gate_inp->buffer == nullptr || ggml_backend_buffer_is_host(mn.ffn_gate_inp->buffer) ||
                    ln.experts.empty() || ln.experts[0]->buffer == nullptr || ggml_backend_buffer_get_type(ln.experts[0]->buffer) != buft_pinned) {
                    continue;
                }
                l.router.gate_inp    = mn.ffn_gate_inp;
                l.router.gate_inp_b  = mn.ffn_gate_inp_b;
                l.router.exp_probs_b = mn.ffn_exp_probs_b;
                l.router.n_pred      = n_pref;
                any = true;
                n_pred_layers++;
            }
            if (!any || model.hparams.no_alloc) {
                continue;
            }
            if (!d.copy) {
                d.copy.reset(ggml_backend_dev_init(dev, nullptr));
            }
            d.ev_fills.reset(ggml_backend_event_new(dev));
            d.ev_pf_prev.reset(ggml_backend_event_new(dev));
            if (!d.copy || !d.ev_fills || !d.ev_pf_prev) {
                throw std::runtime_error("failed to create the MoE prefetch stream");
            }
            for (int32_t i = 0; i < pref_max; ++i) {
                d.ev_pf.emplace_back(ggml_backend_event_new(dev));
                if (!d.ev_pf.back()) {
                    throw std::runtime_error("failed to create the MoE prefetch stream");
                }
            }
        }
        for (group & g : groups) {
            g.pf_unused.assign(g.n_slots, false);
        }
        LLAMA_LOG_INFO("%s: MoE prefetch: %zu layers predict the top %d experts of the next layer, up to %d uploads a layer (LLAMA_MOE_PREFETCH)\n",
            __func__, n_pred_layers, n_pref, pref_max);
    }

    // the staging slots of each device, they overlap for the groups of the device, so a slot fits the experts of its largest layer
    void init_stage(bool no_alloc) {
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            for (size_t il = 0; il < layers.size(); ++il) {
                if (layers[il].ig >= 0 && groups[layers[il].ig].id == (int32_t) id) {
                    layers[il].rank = (int32_t) d.layers.size();
                    d.layers.push_back((int32_t) il);
                }
            }
            if (d.layers.empty()) {
                continue;
            }

            size_t n_tensors = 0;
            for (const group & g : groups) {
                if (g.id == (int32_t) id && g.n_slots > 0) {
                    n_tensors += g.ref.size()*(n_stage + g.layers.size());
                }
            }
            d.ctx_stage = llama_moe_cache_init_ctx(n_tensors);
            ggml_context * ctx = d.ctx_stage.get();

            const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
            std::vector<size_t> sec; // [expert tensor] the largest staging tensor of the device, the slot spacing in a borrowed bank tail
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                group & g = groups[ig];
                if (g.id != (int32_t) id || g.n_slots == 0) {
                    continue;
                }
                g.stage.resize(n_stage);
                for (int32_t is = 0; is < n_stage; ++is) {
                    size_t size = 0;
                    for (size_t ip = 0; ip < g.ref.size(); ++ip) {
                        const ggml_tensor * t = g.ref[ip];
                        // one extra expert at the end, CUDA MMQ can read past the last expert
                        ggml_tensor * st = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], t->ne[2] + 1);
                        GGML_ASSERT(st->nb[2] == t->nb[2]);
                        ggml_format_name(st, "moe_stage.%d.%zu.%s", is, ig, t->name);
                        g.stage[is].push_back(st);
                        const size_t st_size = GGML_PAD(ggml_backend_buft_get_alloc_size(d.buft, st), alignment);
                        size += st_size;
                        sec.resize(std::max(sec.size(), ip + 1), 0);
                        sec[ip] = std::max(sec[ip], st_size);
                    }
                    d.slot_size = std::max(d.slot_size, size);
                }
            }

            std::vector<size_t> start; // [expert tensor] offset of the staging slots in the banks of the lender
            if (borrow && !plan_borrow(d, (int32_t) id, sec, start)) {
                LLAMA_LOG_WARN("%s: %s: the MoE cache is too small to lend %d staging slots, they are allocated separately\n", __func__, ggml_backend_buft_name(d.buft), n_stage);
            }

            if (d.lender < 0) {
                d.buf_stage.reset(ggml_backend_buft_alloc_buffer(d.buft, no_alloc ? 0 : n_stage*d.slot_size));
                if (!d.buf_stage) {
                    throw std::runtime_error("failed to allocate the MoE staging buffers");
                }
            }
            ggml_backend_buffer_t buf = d.lender >= 0 ? d.buf.get() : d.buf_stage.get();
            uint8_t * base = no_alloc ? nullptr : (uint8_t *) ggml_backend_buffer_get_base(buf);
            for (group & g : groups) {
                if (g.stage.empty() || g.id != (int32_t) id) {
                    continue;
                }
                for (int32_t is = 0; is < n_stage; ++is) {
                    size_t offset = is*d.slot_size;
                    for (size_t ip = 0; ip < g.stage[is].size(); ++ip) {
                        ggml_tensor * st = g.stage[is][ip];
                        uint8_t * addr = nullptr;
                        if (no_alloc) {
                            st->buffer = buf;
                        } else {
                            // borrowed: slot is of every tensor position starts at the same place for all groups of the device
                            addr = d.lender >= 0 ? (uint8_t *) groups[d.lender].banks[ip]->data + start[ip] + is*sec[ip] : base + offset;
                            if (ggml_backend_tensor_alloc(buf, st, addr) != GGML_STATUS_SUCCESS) {
                                throw std::runtime_error("failed to allocate the MoE staging buffers");
                            }
                        }
                        offset += GGML_PAD(ggml_backend_buft_get_alloc_size(d.buft, st), alignment);
                    }
                }
            }

            // the matmuls read the n_expert experts of the slot of their layer
            for (int32_t il : d.layers) {
                const layer & l = layers[il];
                const group & g = groups[l.ig];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * st = g.stage[l.rank % n_stage][ip];
                    ggml_tensor * v  = ggml_view_3d(ctx, st, st->ne[0], st->ne[1], st->ne[2] - 1, st->nb[1], st->nb[2], 0);
                    ggml_format_name(v, "moe_stage.%s", l.experts[ip]->name);
                    if (no_alloc) {
                        v->buffer = d.buf_stage.get();
                    } else {
                        ggml_backend_view_init(v);
                    }
                    stage_bindings[l.experts[ip]] = v;
                }
            }

            // as weights, the ops that read the slots run on the device
            if (d.buf_stage) {
                ggml_backend_buffer_set_usage(d.buf_stage.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }

            if (no_alloc) {
                continue;
            }
            if (d.buf_stage) {
                ggml_backend_buffer_clear(d.buf_stage.get(), 0);
            }

            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            d.copy.reset(ggml_backend_dev_init(dev, nullptr));
            d.ev_compute.reset(ggml_backend_event_new(dev));
            if (!d.copy || !d.ev_compute) {
                throw std::runtime_error("failed to create the MoE staging stream");
            }
            for (int32_t is = 0; is < n_stage; ++is) {
                d.ev_ready.emplace_back(ggml_backend_event_new(dev));
                if (!d.ev_ready.back()) {
                    throw std::runtime_error("failed to create the MoE staging stream");
                }
            }
            d.slot_layer.assign(n_stage, -1);

            // an upload from pageable memory blocks the host, which then launches the layer only after the next one is staged
            ggml_backend_buffer_type_t buft_pinned = ggml_backend_dev_host_buffer_type(dev);
            for (int32_t il : d.layers) {
                if (ggml_backend_buffer_get_type(layers[il].experts[0]->buffer) != buft_pinned) {
                    LLAMA_LOG_WARN("%s: the experts of layer %d are not in a pinned host buffer, the staging uploads may block the host\n", __func__, il);
                    break;
                }
            }
        }
    }

    // place the staging slots of device id in the bank tails of its largest group, slot is of tensor position ip at start[ip] + is*sec[ip]
    bool plan_borrow(device & d, int32_t id, const std::vector<size_t> & sec, std::vector<size_t> & start) {
        int32_t best = -1;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            if (g.id != id || g.n_slots == 0) {
                continue;
            }
            if (g.ref.size() != sec.size()) {
                return false;
            }
            if (best < 0 || g.n_slots > groups[best].n_slots) {
                best = (int32_t) ig;
            }
        }
        if (best < 0) {
            return false;
        }
        const group & g = groups[best];
        const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
        int32_t k = 1; // bank slots under the staging slots, including the extra one at the end
        for (size_t ip = 0; ip < sec.size(); ++ip) {
            k = std::max(k, (int32_t) ((n_stage*sec[ip] + g.banks[ip]->nb[2] - 1)/g.banks[ip]->nb[2]));
        }
        // with misses on the CPU, the n_zero slots after the cached ones must stay zero, so the staging slots end before them
        const int32_t end = n_zero > 0 ? g.n_slots : g.n_slots + 1;
        for (; k <= g.n_slots; ++k) {
            const int32_t first = end - k;
            bool fits = true;
            start.assign(sec.size(), 0);
            for (size_t ip = 0; ip < sec.size(); ++ip) {
                start[ip] = GGML_PAD((size_t) first*g.banks[ip]->nb[2], alignment);
                fits = fits && start[ip] + n_stage*sec[ip] <= (size_t) end*g.banks[ip]->nb[2];
            }
            if (fits) {
                d.lender       = best;
                d.borrow_first = first;
                return true;
            }
        }
        return false;
    }

    // take the lender's slots under the staging slots out of the SLRU, called on the compute stream before the first staging of a batch
    // the coldest experts of the lender leave the cache, the hotter ones under the staging slots move to their slots
    void borrow_slots(device & d) {
        group & g = groups[d.lender];
        moe_cache_slru & c = g.slru;
        const int32_t first = d.borrow_first;

        // the slot maps of earlier graphs are uploaded and no stream reads the banks any more
        ggml_backend_synchronize(d.copy.get());
        ggml_backend_synchronize(d.backend);
        d.last_slot = -1;

        std::vector<bool> drop(g.n_slots, false);
        int32_t n_drop = 0;
        for (int seg = 0; seg < 2; ++seg) {
            for (int32_t s = c.head[seg]; s >= 0 && n_drop < g.n_slots - first; s = c.next[s]) {
                drop[s] = true;
                n_drop++;
            }
        }
        GGML_ASSERT(n_drop == g.n_slots - first);

        std::vector<int32_t> src;
        std::vector<int32_t> dst;
        for (int32_t s = first; s < g.n_slots; ++s) {
            if (!drop[s]) {
                src.push_back(s);
            }
        }
        for (int32_t s = 0; s < first; ++s) {
            if (drop[s]) {
                dst.push_back(s);
            }
        }
        GGML_ASSERT(src.size() == dst.size());

        auto evict = [&](int32_t s) {
            if (c.key_of[s] >= 0) {
                c.slot_map[c.key_of[s] / c.n_expert][c.key_of[s] % c.n_expert] = c.empty;
                c.key_of[s] = -1;
                c.n_filled--; // a free slot again: the CPU share waits until the cache is full
                stats_stage.experts_evicted++;
            }
        };
        for (int32_t s : dst) {
            evict(s);
            c.unlink(s);
        }
        size_t n_move = 0;
        for (size_t i = 0; i < src.size(); ++i) {
            const int32_t key = c.key_of[src[i]];
            c.replace(src[i], dst[i]);
            if (key < 0) {
                continue; // an empty slot (cache not full yet): the free slot takes its place, nothing to move
            }
            c.slot_map[key / c.n_expert][key % c.n_expert] = dst[i];
            c.key_of[dst[i]] = key;
            c.key_of[src[i]] = -1;
            src[n_move] = src[i];
            dst[n_move] = dst[i];
            n_move++;
        }
        src.resize(n_move);
        dst.resize(n_move);
        for (int32_t s = first; s < g.n_slots; ++s) {
            if (drop[s]) {
                evict(s);
                c.unlink(s);
            }
        }

        // the moves stay on the compute stream, the staging copies wait for it
        for (size_t i = 0; i < src.size(); ) {
            size_t n = 1;
            while (i + n < src.size() && src[i + n] == src[i] + (int32_t) n && dst[i + n] == dst[i] + (int32_t) n) {
                n++;
            }
            for (ggml_tensor * bank : g.banks) {
                ggml_reset(ctx_views.get());
                ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], src[i]*bank->nb[2]);
                ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], dst[i]*bank->nb[2]);
                ggml_backend_view_init(src_view);
                ggml_backend_view_init(dst_view);
                ggml_backend_tensor_copy_async(d.backend, d.backend, src_view, dst_view);
            }
            i += n;
        }

        stats_stage.experts_moved += src.size();
        stats_stage.borrows++;
        d.borrowed = true;
    }

    // give the borrowed slots back to the cache as free slots, before a batch that the cache serves fills it
    // the compute stream already waited for the last upload of the copy stream
    void return_slots(device & d) {
        moe_cache_slru & c = groups[d.lender].slru;
        for (int32_t s = groups[d.lender].n_slots - 1; s >= d.borrow_first; --s) {
            GGML_ASSERT(c.key_of[s] < 0);
            c.push_front(s);
        }
        std::fill(d.slot_layer.begin(), d.slot_layer.end(), -1);
        d.borrowed = false;
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        for (device & d : devices) {
            if (d.copy) {
                ggml_backend_synchronize(d.copy.get());
            }
        }
        for (dh_device & dd : dh_devs) {
            if (dd.fill) {
                ggml_backend_synchronize(dd.fill.get());
            }
        }
        log_stats();
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
    // 0 tokens (the last layer of a prompt ubatch without outputs, cropped to the output rows) takes the cache path too: the
    // graph then has the topology of one with an output, which the worst case is reserved with, so it does not reallocate
    bool use_cache(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        const layer & l = layers[il];
        return n_tokens >= (zero_rows ? 0 : 1) && n_tokens <= max_batch && std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) <= groups[l.ig].n_slots;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        if (use_cache(il, n_tokens, n_expert_used)) {
            return l.slot_map;
        }
        return n_tokens > 0 ? l.stage_map : nullptr;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    ggml_tensor * get_experts(const ggml_tensor * w, int64_t n_tokens, int64_t n_expert_used) const {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return nullptr;
        }
        return use_cache(it->second.il, n_tokens, n_expert_used) ? it->second.cached : get_staged_experts(w);
    }

    ggml_tensor * get_staged_experts(const ggml_tensor * w) const {
        const auto it = stage_bindings.find(w);
        return it != stage_bindings.end() ? it->second : nullptr;
    }

    bool device_hits(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (dh_mode == 0 || il < 0 || il >= (int32_t) layers.size()) {
            return false;
        }
        const layer & l = layers[il];
        if (l.ig < 0 || l.dh_row < 0 || n_tokens < 1 || n_tokens > dh_max || n_expert_used > this->n_expert_used) {
            return false;
        }
        return use_cache(il, n_tokens, n_expert_used) && groups[l.ig].dh_ok[n_tokens];
    }

    ggml_tensor * get_mapped_experts(const ggml_tensor * w, int64_t n_tokens, int64_t n_expert_used) const {
        const auto it = dh_alias_of.find(w);
        if (it == dh_alias_of.end()) {
            return nullptr;
        }
        return device_hits(bindings.at(w).il, n_tokens, n_expert_used) ? it->second : nullptr;
    }

    // the cached slots of the experts selected by the graph that ran last, logged on the device
    void dh_process(size_t id) {
        dh_device & dd = dh_devs[id];
        int32_t budget = dh_fill_max;
        for (size_t r = 0; r < dd.rows.size(); ++r) {
            const int32_t il = dd.rows[r];
            const layer & l = layers[il];
            group & g = groups[l.ig];
            moe_cache_slru & slru = g.slru;

            const int32_t * ids = dd.log_host.data() + r*dh_log_cap;
            int64_t n = 0;
            while (n < dh_log_cap && ids[n] >= 0) {
                n++;
            }
            if (n == 0) {
                continue;
            }

            size_t expert_bytes = 0;
            for (const ggml_tensor * w : l.experts) {
                expert_bytes += w->nb[2];
            }

            slru.mark_seen(ids, n);
            int32_t * slots = slru.slot_map[il];
            dh_cand.clear();
            for (int32_t e : slru.uniq) {
                if (slots[e] < slru.n_slots) {
                    slru.touch(slots[e]);
                    stats_dh.hits++;
                } else {
                    stats_dh.misses++;
                    stats_dh.bytes_miss += expert_bytes;
                    if (dh_inflight.count(il*slru.n_expert + e) == 0) {
                        dh_cand.push_back(e);
                    }
                }
            }

            // take the slots now: the evicted experts are read from host memory from the next graph on (the device map
            // is uploaded before it), the fill writes a slot that no map points to
            std::sort(dh_cand.begin(), dh_cand.end());
            for (int32_t e : dh_cand) {
                if (budget <= 0) {
                    break;
                }
                const int32_t s = slru.try_victim(il);
                if (s < 0) {
                    break;
                }
                const int32_t key = slru.key_of[s];
                if (key >= 0) {
                    slru.slot_map[key / slru.n_expert][key % slru.n_expert] = slru.empty;
                    dh_map_set(key / slru.n_expert, key % slru.n_expert, -1 - key % slru.n_expert);
                    slru.n_filled--;
                }
                slru.key_of[s] = -1;
                slru.unlink(s);
                dd.pending.push_back({ il, e, s });
                dh_inflight.insert(il*slru.n_expert + e);
                budget--;
            }
        }
    }

    // upload the pending fills on the fill stream, after what the compute stream has queued so far
    void dh_queue(size_t id) {
        dh_device & dd = dh_devs[id];
        if (dd.pending.empty()) {
            return;
        }
        GGML_ASSERT(dd.queued.empty());
        ggml_backend_t fill = dd.fill.get();
        ggml_backend_event_record(dd.ev_after.get(), devices[id].backend);
        ggml_backend_event_wait(fill, dd.ev_after.get());

        // consecutive experts of a layer going to consecutive slots are uploaded together
        std::sort(dd.pending.begin(), dd.pending.end(), [](const dh_fill & a, const dh_fill & b) {
            return a.il != b.il ? a.il < b.il : a.expert < b.expert;
        });
        for (size_t i = 0; i < dd.pending.size(); ) {
            const dh_fill & f = dd.pending[i];
            size_t n = 1;
            while (i + n < dd.pending.size() && dd.pending[i + n].il == f.il && dd.pending[i + n].expert == f.expert + (int32_t) n &&
                    dd.pending[i + n].slot == f.slot + (int32_t) n) {
                n++;
            }
            const layer & l = layers[f.il];
            const group & g = groups[l.ig];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const size_t expert_size = w->nb[2];
                ggml_backend_tensor_set_async(fill, g.banks[ip], (const uint8_t *) w->data + f.expert*expert_size, f.slot*expert_size, n*expert_size);
                stats_dh.bytes_fill += n*expert_size;
            }
            i += n;
        }
        ggml_backend_event_record(dd.ev_fill.get(), fill);
        stats_dh.fills += dd.pending.size();
        dd.queued.swap(dd.pending);
        dd.pending.clear();
    }

    // a fill is done: the expert is cached from the next graph on, unless the cache path uploaded it meanwhile
    void dh_commit(const dh_fill & f) {
        moe_cache_slru & slru = groups[layers[f.il].ig].slru;
        dh_inflight.erase(f.il*slru.n_expert + f.expert);
        if (slru.slot_map[f.il][f.expert] < slru.n_slots) {
            slru.push_front(f.slot);
            stats_dh.dropped++;
            return;
        }
        slru.key_of[f.slot] = f.il*slru.n_expert + f.expert;
        slru.slot_map[f.il][f.expert] = f.slot;
        slru.push_back(f.slot);
        slru.n_filled++;
        dh_map_set(f.il, f.expert, f.slot);
    }

    void dh_begin(ggml_cgraph * gf) {
        if (dh_devs.empty()) {
            return;
        }
        // does this graph log selected experts? (a reused graph is not rebuilt, so look at the graph itself)
        bool logs = false;
        const int n_nodes = gf ? ggml_graph_n_nodes(gf) : 0;
        for (int i = 0; i < n_nodes && !logs; ++i) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            logs = node->op == GGML_OP_CPY && node->view_src != nullptr && dh_logs.count(node->view_src) > 0;
        }

        for (size_t id = 0; id < dh_devs.size(); ++id) {
            dh_device & dd = dh_devs[id];
            if (dd.rows.empty()) {
                continue;
            }
            ggml_backend_t backend = devices[id].backend;
            // read the log first: the synchronization waits for the last graph, not for the fills
            if (dh_logged) {
                ggml_backend_tensor_get_async(backend, dd.log, dd.log_host.data(), 0, ggml_nbytes(dd.log));
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_set_async(backend, dd.log, dd.log_clear.data(), 0, ggml_nbytes(dd.log));
            }
            if (!dd.queued.empty()) {
                ggml_backend_event_wait(backend, dd.ev_fill.get());
                for (const dh_fill & f : dd.queued) {
                    dh_commit(f);
                }
                dd.queued.clear();
            }
            if (dd.resync) {
                dh_rebuild_map(dd);
                dd.resync = false;
            }
            if (dh_logged) {
                dh_process(id);
            }
            if (dd.map_dirty) {
                ggml_backend_tensor_set_async(backend, dd.map, dd.map_host.data(), 0, ggml_nbytes(dd.map));
                dd.map_dirty = false;
            }
            if (dh_mode == 2) {
                dh_queue(id);
            }
        }
        if (dh_logged) {
            stats_dh.graphs++;
        }
        dh_logged = logs;
    }

    void launched() {
        if (dh_mode != 1) {
            return;
        }
        for (size_t id = 0; id < dh_devs.size(); ++id) {
            if (!dh_devs[id].rows.empty()) {
                dh_queue(id);
            }
        }
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    static const ggml_tensor * base_of(const ggml_tensor * t) {
        while (t->view_src != nullptr) {
            t = t->view_src;
        }
        return t;
    }

    // an input of the CPU part of the last layer, it was read from the device with the selected experts
    bool copy_cpu_input(const ggml_tensor * src, ggml_tensor * dst) const {
        if (stash.empty() || dst->buffer == nullptr || !ggml_backend_buffer_is_host(dst->buffer) || src->data == nullptr) {
            return false;
        }
        const ggml_tensor * base = base_of(src);
        const size_t offs = (const uint8_t *) src->data - (const uint8_t *) base->data;
        const size_t size = ggml_nbytes(src);
        for (const stash_entry & e : stash) {
            if (e.base == base && offs >= e.offs && offs + size <= e.offs + e.size) {
                memcpy(dst->data, (const uint8_t *) ggml_backend_buffer_get_base(stash_buf.get()) + e.pos + (offs - e.offs), size);
                return true;
            }
        }
        return false;
    }

    void set_cpu_part(const ggml_tensor * lookup, int32_t n_used, std::vector<const ggml_tensor *> inputs, const ggml_tensor * output) {
        cpu_parts[lookup] = { n_used, std::move(inputs) };
        if (output != nullptr) {
            cpu_outputs.insert(output);
        }
    }

    device * device_of(ggml_backend_t backend) {
        const auto it = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.backend == backend; });
        return it != devices.end() ? &*it : nullptr;
    }

    // the result of a CPU part, the device copies it after the work queued on it, without waiting on the host
    // in one graph, the next CPU part runs after the next slot lookup, which synchronizes its device (and through the activations,
    // the earlier devices), so the result is not overwritten in the CPU compute buffer before it is copied
    // a later graph in flight would not wait for the other devices: then the result goes through buf_out of the device, which is
    // only written again after the next slot lookup of the same device has synchronized it
    bool copy_cpu_output(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
        if (cpu_outputs.count(src) == 0 || src->buffer == nullptr || !ggml_backend_buffer_is_host(src->buffer) ||
            dst->buffer == nullptr || ggml_backend_buffer_is_host(dst->buffer)) {
            return false;
        }
        device * d = device_of(backend);
        if (d == nullptr) {
            return false; // the scheduler copies it synchronously
        }
        const size_t size = ggml_nbytes(src);
        if (!out_buffered) {
            ggml_backend_tensor_set_async(backend, dst, src->data, 0, size);
            d->out_direct = true;
            return true;
        }
        if (!d->out_idle) {
            ggml_backend_synchronize(backend);
        }
        if (!d->buf_out || ggml_backend_buffer_get_size(d->buf_out.get()) < size) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
            d->buf_out.reset(ggml_backend_buft_alloc_buffer(buft ? buft : ggml_backend_cpu_buffer_type(), size));
            GGML_ASSERT(d->buf_out);
        }
        void * data = ggml_backend_buffer_get_base(d->buf_out.get());
        memcpy(data, src->data, size);
        ggml_backend_tensor_set_async(backend, dst, data, 0, size);
        d->out_idle = false;
        return true;
    }

    void begin(bool prev_in_flight, ggml_cgraph * gf) {
        dh_begin(gf);
        stash.clear();
        // large batches read and move the banks on the compute stream without a lookup
        for (device & d : devices) {
            wait_prefetch(d, false);
        }
        // a direct copy of the previous graph reads the CPU compute buffer, which this graph's CPU parts overwrite
        out_buffered = prev_in_flight;
        for (device & d : devices) {
            if (d.out_direct && prev_in_flight) {
                ggml_backend_synchronize(d.backend);
                d.out_idle = true;
            }
            d.out_direct = false;
        }
    }

    // the get_rows that looks up the slots of the selected experts
    static const ggml_tensor * find_lookup(ggml_cgraph * graph, const ggml_tensor * dst) {
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                return node;
            }
        }
        GGML_ABORT("no MoE cache lookup in the split");
    }

    // the device inputs of the CPU part of layer il are read with the selected experts (before read_ids), so the CPU part
    // does not wait for the device; returns the CPU part of the lookup, if the graph has one (cache path only)
    const cpu_part * read_cpu_inputs(ggml_backend_t backend, ggml_cgraph * graph, const ggml_tensor * lookup, const layer & l) {
        const int n_nodes = ggml_graph_n_nodes(graph);
        // the device inputs of the CPU part are read with the selected experts, so the CPU part does not wait for the device
        // the inputs computed in this split are left to the scheduler
        // the graph has a CPU part for this layer if it registered its inputs
        const auto it_cpu = cpu_parts.find(lookup);
        const bool has_cpu = l.cpu_map != nullptr && it_cpu != cpu_parts.end() && !it_cpu->second.inputs.empty();

        stash.clear();
        size_t stash_size = 0;
        if (has_cpu) {
            for (const ggml_tensor * t : it_cpu->second.inputs) {
                const ggml_tensor * base = base_of(t);
                if (base->buffer == nullptr || ggml_backend_buffer_is_host(base->buffer) || t->data == nullptr) {
                    continue;
                }
                bool in_split = false;
                for (int i = 0; i < n_nodes && !in_split; ++i) {
                    in_split = ggml_graph_node(graph, i) == base;
                }
                if (in_split) {
                    continue;
                }
                const size_t offs = (const uint8_t *) t->data - (const uint8_t *) base->data;
                stash.push_back({ base, offs, ggml_nbytes(t), stash_size });
                stash_size += GGML_PAD(ggml_nbytes(t), 64);
            }
            if (!stash_buf || ggml_backend_buffer_get_size(stash_buf.get()) < stash_size) {
                ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
                stash_buf.reset(ggml_backend_buft_alloc_buffer(buft ? buft : ggml_backend_cpu_buffer_type(), 2*stash_size));
                GGML_ASSERT(stash_buf);
            }
            uint8_t * stash_data = (uint8_t *) ggml_backend_buffer_get_base(stash_buf.get());
            for (const stash_entry & e : stash) {
                ggml_backend_tensor_get_async(backend, e.base, stash_data + e.pos, e.offs, e.size);
            }
        }

        return has_cpu ? &it_cpu->second : nullptr;
    }

    // read back the experts selected for the lookup
    void read_ids(ggml_backend_t backend, ggml_cgraph * graph, const ggml_tensor * lookup, int32_t il) {
        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);
    }

    // stage the experts of layer il in slot is on the copy stream of d: all of them, or the used ones
    // the experts in the cache are copied from device memory, the others are uploaded
    void stage(device & d, int32_t il, int32_t is, const std::vector<bool> * used) {
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        const int32_t * slots = g.slru.slot_map[il];
        ggml_backend_t copy = d.copy.get();

        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * st   = g.stage[is][ip];
            ggml_tensor       * bank = g.banks[ip];
            const int64_t n_expert    = w->ne[2];
            const size_t  expert_size = w->nb[2];
            const uint8_t * data = (const uint8_t *) w->data;

            for (int64_t first = 0; first < n_expert; ) {
                if (used && !(*used)[first]) {
                    first++;
                    continue;
                }
                int64_t last = first;
                while (last + 1 < n_expert && (!used || (*used)[last + 1])) {
                    last++;
                }
                for (int64_t e = first; e <= last; ) {
                    int64_t n = 1;
                    if (slots[e] < g.n_slots) {
                        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
                            n++;
                        }
                        ggml_reset(ctx_views.get());
                        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
                        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), st,   st->ne[0],   st->ne[1],   n, st->nb[1],   st->nb[2],   e*st->nb[2]);
                        ggml_backend_view_init(src_view);
                        ggml_backend_view_init(dst_view);
                        ggml_backend_tensor_copy_async(copy, copy, src_view, dst_view);
                        stats_stage.bytes_cache += n*expert_size;
                    } else {
                        while (e + n <= last && slots[e + n] >= g.n_slots) {
                            n++;
                        }
                        ggml_backend_tensor_set_async(copy, st, data + e*expert_size, e*expert_size, n*expert_size);
                        stats_stage.bytes_host += n*expert_size;
                    }
                    e += n;
                }
                // MMQ can read past the last expert of a run: give it the start of the next expert, or zeros after the last one
                const size_t padding = std::min(expert_size, zeros.size());
                if (last + 1 < n_expert) {
                    ggml_backend_tensor_set_async(copy, st, data + (last + 1)*expert_size, (last + 1)*expert_size, padding);
                } else {
                    ggml_backend_tensor_set_async(copy, st, zeros.data(), n_expert*expert_size, padding);
                }
                first = last + 1;
            }
        }

        ggml_backend_event_record(d.ev_ready[is].get(), copy);
        d.last_slot     = is;
        d.slot_layer[is] = used ? -1 : il;
    }

    bool copy_stage(ggml_backend_t backend, int32_t il, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const layer & l = layers[il];
        device & d = devices[groups[l.ig].id];

        GGML_ASSERT(backend == d.backend);

        const ggml_tensor * lookup = find_lookup(graph, dst);
        const int64_t n_expert = l.stage_map->ne[1];
        const bool full = ggml_nelements(lookup->src[1]) >= full_ids*n_expert;

        if (d.lender >= 0 && !d.borrowed) {
            borrow_slots(d);
        }

        // the copy stream does not overwrite a slot or read a bank before the compute stream is done with them
        // the slot of this layer was last read by an earlier layer, and with 2+ slots the slot of the next one too
        ggml_backend_event_record(d.ev_compute.get(), backend);
        ggml_backend_event_wait(d.copy.get(), d.ev_compute.get());

        const int32_t is = l.rank % n_stage;
        if (!full) {
            read_ids(backend, graph, lookup, il);
            used.assign(n_expert, false);
            for (int32_t id : ids) {
                GGML_ASSERT(id >= 0 && id < n_expert);
                used[id] = true;
            }
            stage(d, il, is, &used);
            stats_stage.layers_partial++;
        } else if (d.slot_layer[is] == il) {
            stats_stage.layers_prefetched++;
        } else {
            stage(d, il, is, nullptr);
            stats_stage.layers_waited++;
        }

        // the compute stream waits only for the experts of this layer
        ggml_backend_event_wait(backend, d.ev_ready[is].get());
        if (d.last_slot == is) {
            d.last_slot = -1;
        }

        // the routing of the next layer is not known yet, but a large batch uses almost all of its experts
        if (full && n_stage > 1 && l.rank + 1 < (int32_t) d.layers.size()) {
            const int32_t next    = d.layers[l.rank + 1];
            const int32_t is_next = (l.rank + 1) % n_stage;
            if (d.slot_layer[is_next] != next) {
                stage(d, next, is_next, nullptr);
                stats_stage.prefetches++;
            }
        }

        // the staging map is the identity
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            const auto it_stage = stage_layer_of.find(src);
            if (it_stage != stage_layer_of.end()) {
                return copy_stage(backend, it_stage->second, src, dst, graph);
            }
            return copy_cpu_input(src, dst) || copy_cpu_output(backend, src, dst);
        }
        const int32_t il = it->second;
        layer & l = layers[il];
        group & g = groups[l.ig];
        device & d = devices[g.id];

        GGML_ASSERT(backend == d.backend);

        const ggml_tensor * lookup = find_lookup(graph, dst);
        if (ggml_nelements(lookup) == 0) {
            // no experts selected: nothing to read back or upload, and no synchronization of the graphs in flight
            stash.clear();
            return true;
        }
        const cpu_part * part = read_cpu_inputs(backend, graph, lookup, l);
        const bool has_cpu = part != nullptr;
        const bool has_pred = read_prediction(backend, graph, lookup, il);
        read_ids(backend, graph, lookup, il);
        d.out_idle = true; // read_ids synchronized the device

        // the fills below and the reads of the device part come after the uploads of earlier prefetches, see wait_prefetch_layer
        // for the one of this layer
        wait_prefetch(d, d.pf_layer == il);
        if (!g.pf_unused.empty()) {
            const int32_t * slots = g.slru.slot_map[il];
            for (int32_t e : ids) {
                const int32_t s = e >= 0 && e < g.slru.n_expert ? slots[e] : g.n_slots;
                if (s < g.n_slots && g.pf_unused[s]) {
                    g.pf_unused[s] = false;
                    stats_pf.used++;
                }
            }
        }

        // the fills below may replace experts that the copy stream still reads
        if (d.last_slot >= 0) {
            ggml_backend_event_wait(backend, d.ev_ready[d.last_slot].get());
            d.last_slot = -1;
        }
        if (d.borrowed) {
            return_slots(d);
        }

        if (trace != nullptr) {
            const auto it_cpu = cpu_parts.find(lookup);
            write_trace(il, it_cpu != cpu_parts.end() ? it_cpu->second.n_used : n_expert_used);
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t share_q = has_cpu && g.slru.n_filled == g.slru.n_slots ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), has_cpu ? part->n_used : 1, n_zero, n_cpu, rng, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (!dh_devs.empty()) {
            dh_devs[g.id].resync = true; // the device map follows at the next graph
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
            }
        }

        if (!g.pf_unused.empty()) {
            for (const auto & f : fills) {
                if (g.pf_unused[f.slot]) {
                    g.pf_unused[f.slot] = false;
                    stats_pf.wasted++;
                }
            }
        }
        wait_prefetch_layer(d, il);

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        const size_t bytes = upload(backend, l, g, fills);

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        // the uploads and every read of the banks (also copy_experts) are on the compute stream of the device, so with pipeline
        // parallelism a graph still in flight reads a slot before a later graph evicts it; the staging copy stream (LLAMA_MOE_ASYNC)
        // reads the banks only after waiting for ev_compute, and the fills above wait for its last upload (last_slot)
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        if (has_pred) {
            prefetch(backend, d, il);
        }

        return true;
    }

    // upload the experts of layer l to their slots on the stream of backend, consecutive experts going to consecutive slots together
    static size_t upload(ggml_backend_t backend, const layer & l, const group & g, const std::vector<moe_cache_slru::fill> & fills) {
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }
        return bytes;
    }

    // read back the experts of the next layer predicted for the lookup with its selected experts (the same synchronization)
    // returns false if the lookup has no prediction or it is computed in the split of the lookup
    bool read_prediction(ggml_backend_t backend, ggml_cgraph * graph, const ggml_tensor * lookup, int32_t il) {
        const auto it = predictions.find(lookup);
        if (it == predictions.end() || it->second.pred == nullptr || layers[il].router.n_pred == 0) {
            return false;
        }
        const ggml_tensor * pred = it->second.pred;
        if (pred->buffer == nullptr || pred->data == nullptr || pred->type != GGML_TYPE_I32 || !ggml_is_contiguous(pred)) {
            return false;
        }
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            if (ggml_graph_node(graph, i) == pred) {
                return false;
            }
        }
        pred_ids.resize(ggml_nelements(pred));
        ggml_backend_tensor_get_async(backend, pred, pred_ids.data(), 0, ggml_nbytes(pred));
        return true;
    }

    // upload the predicted experts of the layer after il that are not cached on the copy stream, after the fills of layer il,
    // so they overlap the device part of layer il, its CPU part and the attention of the next layer
    // no stream reads the victims: the device part of il reads its selected experts only (spared), earlier reads were synchronized
    // by read_ids; the next cached lookup of the device and the next graph wait for ev_pf before they fill or read a slot
    void prefetch(ggml_backend_t backend, device & d, int32_t il) {
        const int32_t il_next = il + 1;
        const layer & l  = layers[il];
        const layer & ln = layers[il_next];
        group & gn = groups[ln.ig];

        // rank major: the best guess of every token first
        const int64_t k = l.router.n_pred;
        const int64_t n_tokens = (int64_t) pred_ids.size() / k;
        ids_pf.resize(pred_ids.size());
        for (int64_t it = 0; it < n_tokens; ++it) {
            for (int64_t r = 0; r < k; ++r) {
                ids_pf[r*n_tokens + it] = pred_ids[it*k + r];
            }
        }

        pf_fills.clear();
        gn.slru.prefetch(ln.ig == l.ig ? il : -1, il_next, ids_pf.data(), ids_pf.size(), (size_t) pref_max, pf_fills);
        stats_pf.layers++;
        if (pf_fills.empty()) {
            return;
        }
        for (const auto & f : pf_fills) {
            if (gn.pf_unused[f.slot]) {
                stats_pf.wasted++;
            }
            gn.pf_unused[f.slot] = true;
        }

        ggml_backend_t copy = d.copy.get();
        // uploads of the last prefetch that the compute stream did not wait for end before this one
        if (d.pf_pending) {
            ggml_backend_event_record(d.ev_pf_prev.get(), copy);
            d.pf_prev_pending = true;
        }
        // the fills of layer il go first on the link
        ggml_backend_event_record(d.ev_fills.get(), backend);
        ggml_backend_event_wait(copy, d.ev_fills.get());
        d.pf_slot.clear();
        for (size_t i = 0; i < pf_fills.size(); ++i) {
            stats_pf.bytes += upload(copy, ln, gn, { pf_fills[i] });
            ggml_backend_event_record(d.ev_pf[i].get(), copy);
            d.pf_slot.push_back(pf_fills[i].slot);
        }
        stats_pf.issued += pf_fills.size();
        d.pf_layer   = il_next;
        d.pf_pending = true;
    }

    // the compute stream of d waits for the uploads of the earlier prefetches, and for those of the last one unless keep_last
    void wait_prefetch(device & d, bool keep_last) {
        if (d.pf_prev_pending) {
            ggml_backend_event_wait(d.backend, d.ev_pf_prev.get());
            d.pf_prev_pending = false;
        }
        if (d.pf_pending && !keep_last) {
            ggml_backend_event_wait(d.backend, d.ev_pf[d.pf_slot.size() - 1].get());
            d.pf_pending = false;
        }
    }

    // after the plan of layer il, before its fills: the compute stream waits for the prefetched uploads up to the last one that the
    // layer reads (selected) or overwrites (filled); the later ones (wrong guesses) are still read by no one and may go on
    void wait_prefetch_layer(device & d, int32_t il) {
        if (!d.pf_pending || d.pf_layer != il) {
            return;
        }
        const group & g = groups[layers[il].ig];
        const moe_cache_slru & c = g.slru;
        int32_t last = -1;
        for (int32_t i = 0; i < (int32_t) d.pf_slot.size(); ++i) {
            const int32_t s = d.pf_slot[i];
            const int32_t key = c.key_of[s];
            bool needed = key >= 0 && key / c.n_expert == il && c.seen[key % c.n_expert] == c.seen_gen;
            for (size_t j = 0; j < fills.size() && !needed; ++j) {
                needed = fills[j].slot == s;
            }
            if (needed) {
                last = i;
            }
        }
        if (last >= 0) {
            ggml_backend_event_wait(d.backend, d.ev_pf[last].get());
        }
        if (last == (int32_t) d.pf_slot.size() - 1) {
            d.pf_pending = false;
        } else {
            d.pf_layer = -1; // the rest is waited for by the next lookup of the device or the next prefetch
        }
    }

    bool get_next_router(int32_t il, int64_t n_tokens, int64_t n_expert_used, next_router & r) const {
        if (n_pref == 0 || il < 0 || il + 1 >= (int32_t) layers.size() || layers[il].ig < 0 || layers[il].router.n_pred == 0 ||
            !use_cache(il, n_tokens, n_expert_used) || layers[il + 1].ig < 0 || !use_cache(il + 1, n_tokens, n_expert_used)) {
            return false;
        }
        r = layers[il].router;
        return true;
    }

    void set_prediction(const ggml_tensor * lookup, const ggml_tensor * pred) {
        predictions[lookup] = { pred };
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        const stage_stats & ss = stats_stage;
        if (ss.layers_prefetched + ss.layers_waited + ss.layers_partial > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: staged layers: %zu prefetched, %zu waited for, %zu partial, %zu prefetches, %.2f MiB uploaded, %.2f MiB from the cache\n",
                ss.layers_prefetched, ss.layers_waited, ss.layers_partial, ss.prefetches, ss.bytes_host/1024.0/1024.0, ss.bytes_cache/1024.0/1024.0);
        }
        if (stats_pf.layers > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: prefetch: %zu predictions, %zu experts uploaded ahead (%.2f MiB), %zu selected (%.1f%%), %zu evicted unused\n",
                stats_pf.layers, stats_pf.issued, stats_pf.bytes/1024.0/1024.0, stats_pf.used, stats_pf.issued ? 100.0*stats_pf.used/stats_pf.issued : 0.0, stats_pf.wasted);
        }
        if (ss.borrows > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: staging borrowed the cache %zu times: %zu experts moved, %zu evicted\n", ss.borrows, ss.experts_moved, ss.experts_evicted);
        }
        const dh_stats & ds = stats_dh;
        if (ds.hits + ds.misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: device hits: %zu graphs, hits = %zu, misses = %zu read from host memory (%.2f MiB), hit rate = %.2f%%, "
                "fills = %zu (%.2f MiB), %zu dropped\n", ds.graphs, ds.hits, ds.misses, ds.bytes_miss/1024.0/1024.0,
                100.0*ds.hits/(ds.hits + ds.misses), ds.fills, ds.bytes_fill/1024.0/1024.0, ds.dropped);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
    pimpl(new impl(model, backends, bufts, size, cpu_share)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_experts(w, n_tokens, n_expert_used);
}

bool llama_moe_cache::uses_cache(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return il >= 0 && il < (int32_t) pimpl->layers.size() && pimpl->layers[il].ig >= 0 && pimpl->use_cache(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_staged_experts(const ggml_tensor * w) const {
    return pimpl->get_staged_experts(w);
}

float llama_moe_cache::cpu_share() const {
    return pimpl->cpu_share;
}

ggml_tensor * llama_moe_cache::get_cpu_map(int32_t il) const {
    return il >= 0 && il < (int32_t) pimpl->layers.size() ? pimpl->layers[il].cpu_map : nullptr;
}

void llama_moe_cache::set_cpu_part(const ggml_tensor * lookup, int32_t n_used, std::vector<const ggml_tensor *> inputs, const ggml_tensor * output) {
    pimpl->set_cpu_part(lookup, n_used, std::move(inputs), output);
}

bool llama_moe_cache::get_next_router(int32_t il, int64_t n_tokens, int64_t n_expert_used, next_router & r) const {
    return pimpl->get_next_router(il, n_tokens, n_expert_used, r);
}

void llama_moe_cache::set_prediction(const ggml_tensor * lookup, const ggml_tensor * pred) {
    pimpl->set_prediction(lookup, pred);
}

void llama_moe_cache::begin(bool prev_in_flight, ggml_cgraph * graph) {
    pimpl->begin(prev_in_flight, graph);
}

void llama_moe_cache::launched() {
    pimpl->launched();
}

bool llama_moe_cache::device_hits(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->device_hits(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_device_map(int32_t il) const {
    return il >= 0 && il < (int32_t) pimpl->layers.size() ? pimpl->layers[il].dh_map : nullptr;
}

ggml_tensor * llama_moe_cache::get_device_log(int32_t il) const {
    return il >= 0 && il < (int32_t) pimpl->layers.size() ? pimpl->layers[il].dh_log : nullptr;
}

ggml_tensor * llama_moe_cache::get_mapped_experts(const ggml_tensor * w, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_mapped_experts(w, n_tokens, n_expert_used);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
        if (d.buf_stage) {
            res[d.buft] += pimpl->n_stage*d.slot_size;
        }
    }
    for (const auto & dd : pimpl->dh_devs) {
        if (dd.buf) {
            res[ggml_backend_buffer_get_type(dd.buf.get())] += ggml_backend_buffer_get_size(dd.buf.get());
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
