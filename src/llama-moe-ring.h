#pragma once

// LLAMA_MOE_RING: bookkeeping of the expert ring of one device, no ggml calls (tests/test-moe-ring.cpp)
//
// A graph stages the experts of a device's layers in K layer-sized slots, slot = rank of the layer on its device % K (fixed by
// the graph). The ring streams every expert of the staged layers that is not in the cache on the device's copy stream, in layer
// order and without the routing, as far ahead as the slots allow: the copy into a slot waits for the compute stream to be done
// with the slot's previous layer. Partial layers (batches below the full-batch rule) need their routing, so they are staged at
// their own layer and the ring does not pass them.

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_moe_ring_sched {
    struct entry {
        int32_t il;
        int32_t slot;
        bool    full;
        bool    issued = false;
    };

    int32_t k = 0;
    std::vector<entry>   plan;
    size_t               next_issue   = 0;
    size_t               next_consume = 0;
    std::vector<int32_t> holder;    // [k] plan index whose experts the slot holds or receives, -1 if the slot is free
    std::vector<bool>    must_wait; // [k] the next copy into the slot waits for the release event of its last reader
    int32_t              consumed = -1; // slot of the last layer read, released at the next step

    void init(int32_t n_slots) {
        k = n_slots;
        start({}, true);
    }

    // a new graph, after the last one released its slots (see release): fence = the copy stream waited for everything queued on
    // the compute stream, else each slot's next copy waits for the release event of its last reader
    void start(std::vector<entry> p, bool fence) {
        plan = std::move(p);
        next_issue   = 0;
        next_consume = 0;
        holder.assign(k, -1);
        if (fence || (int32_t) must_wait.size() != k) {
            must_wait.assign(k, false);
        }
        consumed = -1;
    }

    bool active() const {
        return next_consume < plan.size();
    }

    // the plan index to issue next, -1 if none can be issued ahead
    int64_t can_issue() const {
        if (next_issue >= plan.size()) {
            return -1;
        }
        const entry & e = plan[next_issue];
        if (!e.full || holder[e.slot] >= 0) {
            return -1;
        }
        return (int64_t) next_issue;
    }

    // the copies of plan entry i are queued, returns whether they wait for the release event of the slot
    bool issue(size_t i) {
        entry & e = plan[i];
        holder[e.slot] = (int32_t) i;
        e.issued   = true;
        next_issue = i + 1;
        const bool w = must_wait[e.slot];
        must_wait[e.slot] = false;
        return w;
    }

    // at a staged layer: the slot of the layer read before is released, returns it (its release event is recorded now), -1 if none
    int32_t release() {
        const int32_t s = consumed;
        if (s >= 0) {
            holder[s]    = -1;
            must_wait[s] = true;
            consumed     = -1;
        }
        return s;
    }

    // the plan index of layer il at its callback, -1 if it is not the next planned layer (the ring is then dropped for the graph)
    int64_t at_layer(int32_t il) const {
        return next_consume < plan.size() && plan[next_consume].il == il ? (int64_t) next_consume : -1;
    }

    // the layer of plan entry i reads its slot from now on
    void consume(size_t i) {
        consumed     = plan[i].slot;
        next_consume = i + 1;
    }

    void drop() {
        plan.clear();
        next_issue   = 0;
        next_consume = 0;
    }
};

// the copies that fill a slot with the experts of one layer: runs of experts uploaded from host memory, cached experts copied from
// their cache slot, and the bytes MMQ reads past the end of a run (the next expert's start, or zeros after the last expert)
struct llama_moe_ring_copies {
    struct run {
        int32_t first;
        int32_t n;
    };
    std::vector<run>     upload;
    std::vector<int32_t> cached;
    std::vector<int32_t> pad_next; // experts whose first bytes follow a run that is not continued
    bool                 pad_end = false;

    // slots[e] < n_slots: expert e is cached; used == nullptr: all experts; merge: a gap of at most merge cached experts between two
    // uploaded runs is uploaded too (fewer copies queued, more bytes)
    void plan(const int32_t * slots, int32_t n_slots, int32_t n_expert, const std::vector<bool> * used, int32_t merge) {
        upload.clear();
        cached.clear();
        pad_next.clear();
        pad_end = false;

        auto covered = [&](int32_t e) { return used == nullptr || (*used)[e]; };
        for (int32_t first = 0; first < n_expert; ) {
            if (!covered(first)) {
                first++;
                continue;
            }
            int32_t last = first;
            while (last + 1 < n_expert && covered(last + 1)) {
                last++;
            }
            // within [first, last]: uploaded runs of uncached experts, cached gaps up to merge experts joined into them
            size_t run0 = upload.size();
            for (int32_t e = first; e <= last; ) {
                if (slots[e] < n_slots) {
                    cached.push_back(e);
                    e++;
                    continue;
                }
                int32_t n = 1;
                while (e + n <= last && slots[e + n] >= n_slots) {
                    n++;
                }
                upload.push_back({ e, n });
                e += n;
            }
            if (merge > 0 && upload.size() - run0 > 1) {
                size_t w = run0;
                for (size_t r = run0 + 1; r < upload.size(); ++r) {
                    run & prev = upload[w];
                    const int32_t gap = upload[r].first - (prev.first + prev.n);
                    if (gap <= merge) {
                        prev.n = upload[r].first + upload[r].n - prev.first;
                    } else {
                        upload[++w] = upload[r];
                    }
                }
                upload.resize(w + 1);
                // the cached experts inside the merged runs are uploaded
                size_t c = 0;
                size_t ir = run0;
                for (int32_t e : cached) {
                    while (ir < upload.size() && upload[ir].first + upload[ir].n <= e) {
                        ir++;
                    }
                    if (e < first || ir >= upload.size() || e < upload[ir].first) {
                        cached[c++] = e;
                    }
                }
                cached.resize(c);
            }
            if (last + 1 < n_expert) {
                pad_next.push_back(last + 1);
            } else {
                pad_end = true;
            }
            first = last + 1;
        }
    }
};
