// LLAMA_MOE_RING bookkeeping: the slot schedule (llama_moe_ring_sched) driven the way llama_moe_cache drives it, and the copy plan
// of a layer (llama_moe_ring_copies), against invariants:
// - a copy into a slot is queued only after the layer that read the slot before has been released (its matmuls queued) and, when
//   the copy is not behind a fence, waits for that release event
// - every staged layer finds its copies queued at its callback, in plan order, and a full layer is queued ahead when a slot allows
// - the copy plan covers every selected expert exactly once: uploaded if not cached, uploaded or copied from the cache if cached
#include "../src/llama-moe-ring.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int fails = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fails++; if (fails < 20) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

// a device: the ring driven like ring_begin_device / ring_copy_stage / ring_end_device
struct sim_device {
    llama_moe_ring_sched rs;
    int32_t k;
    // per slot: the plan entry (graph, index) last queued into it, whether its reader released it (event recorded), and whether
    // the release event has been waited for by a later copy (or a fence covered it)
    struct slot_state {
        int64_t graph = -1;
        int64_t index = -1;
        bool    read     = false;
        bool    released = true;
        bool    waited   = true;
    };
    std::vector<slot_state> slots;
    int64_t graph = -1;
    bool ring_prev  = false;
    bool ring_fence = false;
    size_t ahead = 0, at_layer = 0, partial = 0;

    explicit sim_device(int32_t k) : k(k) {
        rs.init(k);
        slots.resize(k);
    }

    void fill(size_t i) {
        const auto e = rs.plan[i];
        CHECK(!e.issued, "entry issued twice");
        CHECK(rs.holder[e.slot] < 0, "slot %d busy", e.slot);
        slot_state & s = slots[e.slot];
        // the slot's last reader must be done before the copy overwrites it (a slot never read may be overwritten: copies are in order)
        CHECK(!s.read || s.released, "graph %lld entry %zu: slot %d not released by its reader", (long long) graph, i, e.slot);
        const bool w = rs.issue(i);
        if (s.read && !s.waited) {
            CHECK(w, "graph %lld entry %zu: copy into slot %d does not wait for its release event", (long long) graph, i, e.slot);
        }
        s = { graph, (int64_t) i, false, false, false };
    }

    void issue() {
        for (int64_t i = rs.can_issue(); i >= 0; i = rs.can_issue()) {
            fill((size_t) i);
            ahead++;
        }
    }

    void begin(std::vector<llama_moe_ring_sched::entry> plan) {
        graph++;
        if (plan.empty()) {
            rs.drop();
            ring_prev = false;
            return;
        }
        const bool fence = !ring_prev || ring_fence;
        if (fence) {
            // the copy stream waits for the whole compute stream
            for (slot_state & s : slots) {
                s.released = true;
                s.waited   = true;
            }
        }
        rs.start(std::move(plan), fence);
        ring_prev  = true;
        ring_fence = false;
        issue();
    }

    void release() {
        const int32_t s = rs.release();
        if (s >= 0) {
            CHECK(slots[s].read, "released slot %d was not read", s);
            slots[s].released = true;
        }
    }

    // returns false if the ring does not serve the layer
    bool at_layer_cb(int32_t il) {
        if (!rs.active()) {
            return false;
        }
        const int64_t i = rs.at_layer(il);
        if (i < 0) {
            rs.drop();
            ring_fence = true;
            return false;
        }
        release();
        const auto e = rs.plan[i];
        if (!e.issued) {
            fill((size_t) i);
            if (e.full) {
                at_layer++;
            } else {
                partial++;
            }
        }
        issue();
        slot_state & s = slots[e.slot];
        CHECK(s.graph == graph && s.index == i, "layer %d reads slot %d holding graph %lld entry %lld", il, e.slot, (long long) s.graph, (long long) s.index);
        s.read = true;
        rs.consume((size_t) i);
        return true;
    }

    void end() {
        release();
    }
};

static void test_sched(std::mt19937 & rng) {
    for (int iter = 0; iter < 2000; ++iter) {
        const int32_t k        = 2 + rng() % 3;
        const int32_t n_layers = 1 + rng() % 14;
        sim_device d(k);
        const int n_graphs = 1 + rng() % 8;
        for (int g = 0; g < n_graphs; ++g) {
            // a graph stages the device's layers (some skipped: the last layer on the output rows), the last ones may be partial;
            // sometimes a graph without staged layers (decode), sometimes a layer missing at the callbacks (plan dropped)
            std::vector<llama_moe_ring_sched::entry> plan;
            const bool decode = rng() % 6 == 0;
            const bool partial_tail = rng() % 4 == 0;
            for (int32_t r = 0; r < n_layers && !decode; ++r) {
                if (r == n_layers - 1 && rng() % 3 == 0) {
                    continue;
                }
                llama_moe_ring_sched::entry e;
                e.il   = 100 + r;
                e.slot = r % k;
                e.full = !(partial_tail && r >= n_layers - 2) && rng() % 20 != 0;
                plan.push_back(e);
            }
            const std::vector<llama_moe_ring_sched::entry> layers = plan;
            d.begin(plan);
            const bool skip = rng() % 10 == 0 && layers.size() > 1;
            const size_t skip_at = skip ? rng() % layers.size() : layers.size();
            bool dropped = false;
            for (size_t i = 0; i < layers.size(); ++i) {
                if (i == skip_at) {
                    continue;
                }
                const bool served = d.at_layer_cb(layers[i].il);
                if (!dropped) {
                    CHECK(served || i > skip_at, "layer %zu not served", i);
                }
                if (!served) {
                    dropped = true;
                    // the per-layer staging: the copy stream waits for the compute stream at this layer (ev_compute) and writes the slot
                    for (sim_device::slot_state & s : d.slots) {
                        s.released = true;
                        s.waited   = true;
                    }
                    sim_device::slot_state & s = d.slots[layers[i].slot];
                    s = { d.graph, -2, true, false, true };
                }
            }
            if (rng() % 5 == 0) {
                // a cache lookup on the device (output rows of the last layer): fills, the ring stops for the graph
                d.rs.drop();
                d.ring_fence = true;
            }
            d.end();
        }
    }
}

static void test_lookahead() {
    // 12 layers, K = 2, all full: after the first graph's first two, every layer is queued one layer ahead (none at its own layer)
    sim_device d(2);
    for (int g = 0; g < 4; ++g) {
        std::vector<llama_moe_ring_sched::entry> plan;
        for (int32_t r = 0; r < 12; ++r) {
            plan.push_back({ r, r % 2, true });
        }
        d.begin(plan);
        for (int32_t r = 0; r < 12; ++r) {
            d.at_layer_cb(r);
        }
        d.end();
    }
    CHECK(d.at_layer == 0 && d.ahead == 48, "lookahead: %zu ahead, %zu at their layer", d.ahead, d.at_layer);

    // 13 layers, K = 2: layer 0 of the next graph has the slot of layer 12, it is still queued at begin, after the release
    sim_device o(2);
    for (int g = 0; g < 3; ++g) {
        std::vector<llama_moe_ring_sched::entry> plan;
        for (int32_t r = 0; r < 13; ++r) {
            plan.push_back({ r, r % 2, true });
        }
        o.begin(plan);
        for (int32_t r = 0; r < 13; ++r) {
            o.at_layer_cb(r);
        }
        o.end();
    }
    CHECK(o.at_layer == 0, "odd layers: %zu at their layer", o.at_layer);
}

static void test_copies(std::mt19937 & rng) {
    for (int iter = 0; iter < 5000; ++iter) {
        const int32_t n_expert = 1 + rng() % 600;
        const int32_t n_slots  = 1 + rng() % 4000;
        const double  p_cached = (rng() % 100)/100.0;
        std::vector<int32_t> slots(n_expert);
        for (int32_t e = 0; e < n_expert; ++e) {
            slots[e] = (rng() % 1000)/1000.0 < p_cached ? (int32_t) (rng() % n_slots) : n_slots + (int32_t) (rng() % 3);
        }
        std::vector<bool> used(n_expert);
        const bool partial = rng() % 2 == 0;
        for (int32_t e = 0; e < n_expert; ++e) {
            used[e] = rng() % 3 != 0;
        }
        const int32_t merge = rng() % 3 == 0 ? (int32_t) (rng() % 4) : 0;

        llama_moe_ring_copies cp;
        cp.plan(slots.data(), n_slots, n_expert, partial ? &used : nullptr, merge);
        auto covered = [&](int32_t e) { return !partial || used[e]; };

        std::vector<int> count(n_expert, 0);
        int32_t prev_end = -1;
        for (const auto & r : cp.upload) {
            CHECK(r.n > 0 && r.first >= 0 && r.first + r.n <= n_expert, "bad run");
            CHECK(r.first >= prev_end, "runs out of order or overlapping");
            prev_end = r.first + r.n;
            for (int32_t e = r.first; e < r.first + r.n; ++e) {
                CHECK(covered(e), "run uploads an expert that is not selected");
                count[e]++;
            }
        }
        for (int32_t e : cp.cached) {
            CHECK(slots[e] < n_slots, "expert %d copied from the cache but not cached", e);
            count[e]++;
        }
        size_t n_cached_uploaded = 0;
        for (int32_t e = 0; e < n_expert; ++e) {
            CHECK(count[e] == (covered(e) ? 1 : 0), "expert %d copied %d times", e, count[e]);
        }
        for (const auto & r : cp.upload) {
            for (int32_t e = r.first; e < r.first + r.n; ++e) {
                n_cached_uploaded += slots[e] < n_slots;
            }
        }
        if (merge == 0) {
            CHECK(n_cached_uploaded == 0, "a cached expert is uploaded without merging");
        }
        // padding: after each maximal selected run, the next expert's start or zeros at the end
        size_t n_pad = 0;
        bool want_end = false;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (covered(e) && (e + 1 == n_expert || !covered(e + 1))) {
                if (e + 1 == n_expert) {
                    want_end = true;
                } else {
                    CHECK(n_pad < cp.pad_next.size() && cp.pad_next[n_pad] == e + 1, "missing padding after expert %d", e);
                    n_pad++;
                }
            }
        }
        CHECK(n_pad == cp.pad_next.size(), "extra padding");
        CHECK(cp.pad_end == want_end, "pad_end");
    }

    // the full layer of a mostly cached device: one run per uncached stretch, merged with merge = 1
    std::vector<int32_t> slots = { 0, 9, 1, 9, 9, 2, 9, 3, 4, 9 };
    llama_moe_ring_copies cp;
    cp.plan(slots.data(), 9, 10, nullptr, 0);
    CHECK(cp.upload.size() == 4 && cp.cached.size() == 5 && cp.pad_end && cp.pad_next.empty(), "example: %zu runs, %zu cached", cp.upload.size(), cp.cached.size());
    cp.plan(slots.data(), 9, 10, nullptr, 1);
    CHECK(cp.upload.size() == 2 && cp.cached.size() == 3, "example merge 1: %zu runs, %zu cached", cp.upload.size(), cp.cached.size());
}

// LLAMA_MOE_RING_INPLACE: every used expert resolves to its own weights (cache slot holding it, staging place it is copied to, or
// the host experts), no slot is read in place when MMQ's read past its end would leave the occupied slots, the staging slot holds
// at most cap experts, host memory only on overflow (busiest experts staged first), never the last expert with read_past
static void test_inplace(std::mt19937 & rng) {
    size_t n_plans = 0, n_host = 0, n_refused = 0;
    for (int it = 0; it < 20000; ++it) {
        const int32_t n_expert = 1 + (int32_t) (rng() % (it < 1000 ? 16 : 600));
        const int32_t n_slots  = (int32_t) (rng() % (2*n_expert + 2));
        // slots: some experts of this layer cached, the other slots empty or holding other layers' experts
        std::vector<int32_t> key_of(n_slots, -1);
        std::vector<int32_t> slots(n_expert, n_slots + (int32_t) (rng() % 3)); // >= n_slots: not cached
        std::vector<int32_t> free_slots(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            free_slots[s] = s;
        }
        std::shuffle(free_slots.begin(), free_slots.end(), rng);
        const int32_t share = (int32_t) (rng() % 101);
        size_t next = 0;
        for (int32_t e = 0; e < n_expert && next < free_slots.size(); ++e) {
            if ((int32_t) (rng() % 100) < share) {
                slots[e] = free_slots[next++];
                key_of[slots[e]] = e;
            }
        }
        for (; next < free_slots.size(); ++next) {
            if (rng() % 2) {
                key_of[free_slots[next]] = 100000 + (int32_t) next; // another layer's expert
            }
        }
        const bool partial   = rng() % 2;
        const bool counted   = partial && rng() % 2;
        const bool can_host  = rng() % 4 != 0;
        const bool read_past = rng() % 2;
        const int32_t cap    = (int32_t) (rng() % (n_expert + 1));
        std::vector<bool> used(n_expert, true);
        std::vector<int32_t> count(n_expert, 0);
        if (partial) {
            for (int32_t e = 0; e < n_expert; ++e) {
                used[e]  = rng() % 3 != 0;
                count[e] = used[e] ? 1 + (int32_t) (rng() % 50) : 0;
            }
        }

        llama_moe_ring_inplace ip;
        const bool ok = ip.plan(slots.data(), n_slots, key_of.data(), n_expert, partial ? &used : nullptr, counted ? &count : nullptr,
            cap, can_host, read_past);

        // the misses the rule must stage or read from host memory
        auto in_place = [&](int32_t e) {
            const int32_t s = slots[e];
            return s < n_slots && (!read_past || (s + 1 < n_slots && key_of[s + 1] >= 0));
        };
        int32_t n_miss = 0, n_hits = 0;
        bool last_miss = false;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (used[e]) {
                if (in_place(e)) {
                    n_hits++;
                } else {
                    n_miss++;
                    last_miss = last_miss || e == n_expert - 1;
                }
            }
        }
        if (!ok) {
            n_refused++;
            CHECK(n_miss > cap && (!can_host || (cap == 0 && read_past && last_miss)), "refused without cause: %d misses, cap %d", n_miss, cap);
            continue;
        }
        n_plans++;
        n_host += !ip.host.empty();
        CHECK(ip.n_hit == n_hits, "hits %d vs %d", ip.n_hit, n_hits);
        CHECK((int32_t) ip.staged.size() <= cap, "staged %zu > cap %d", ip.staged.size(), cap);
        CHECK((int32_t) (ip.staged.size() + ip.host.size()) == n_miss, "misses %d, staged %zu + host %zu", n_miss, ip.staged.size(), ip.host.size());
        CHECK(n_miss <= cap ? ip.host.empty() : (int32_t) ip.staged.size() == cap, "host used without overflow / staging not full");
        CHECK(std::is_sorted(ip.staged.begin(), ip.staged.end()) && std::adjacent_find(ip.staged.begin(), ip.staged.end()) == ip.staged.end(), "staged order");
        CHECK(ip.host.empty() || can_host, "host memory without can_host");
        // every used expert reads its own weights: bank slot s holds key_of[s], staging place p holds staged[p], host holds all
        for (int32_t e = 0; e < n_expert; ++e) {
            if (!used[e]) {
                continue;
            }
            const int32_t v = ip.map[e];
            if (v >= 0) {
                CHECK(v < n_slots && key_of[v] == e, "expert %d read from slot %d holding %d", e, v, v < n_slots ? key_of[v] : -9);
                CHECK(!read_past || (v + 1 < n_slots && key_of[v + 1] >= 0), "slot %d read in place, read past lands outside the experts", v);
            } else if (-1 - v < cap) {
                const int32_t p = -1 - v;
                CHECK(p < (int32_t) ip.staged.size() && ip.staged[p] == e, "expert %d read from place %d", e, p);
            } else {
                CHECK(std::binary_search(ip.host.begin(), ip.host.end(), e), "expert %d read from host memory but not listed", e);
                CHECK(!(read_past && e == n_expert - 1), "the last expert read from host memory with read past");
            }
        }
        if (counted && !ip.host.empty()) {
            int32_t min_staged = INT32_MAX, max_host = 0;
            for (int32_t e : ip.staged) {
                if (!(read_past && e == n_expert - 1)) {
                    min_staged = std::min(min_staged, count[e]);
                }
            }
            for (int32_t e : ip.host) {
                max_host = std::max(max_host, count[e]);
            }
            CHECK(min_staged >= max_host, "a busier expert (%d tokens) read from host memory than one staged (%d)", max_host, min_staged);
        }
    }
    CHECK(n_plans > 5000 && n_host > 1000 && n_refused > 100, "coverage: %zu plans, %zu with host reads, %zu refused", n_plans, n_host, n_refused);

    // Flash-Next down projection: slot 1 is followed by an empty slot, so its expert (3) is staged with read past
    std::vector<int32_t> slots  = { 0, 8, 3, 1, 8 };
    std::vector<int32_t> key_of = { 0, 3, -1, 2, 7, -1, 5, 6 };
    llama_moe_ring_inplace ip;
    CHECK(ip.plan(slots.data(), 8, key_of.data(), 5, nullptr, nullptr, 3, true, true), "example refused");
    CHECK(ip.map[0] == 0 && ip.map[2] == 3 && ip.staged == std::vector<int32_t>({ 1, 3, 4 }) && ip.host.empty(), "example read past");
    CHECK(ip.plan(slots.data(), 8, key_of.data(), 5, nullptr, nullptr, 3, true, false), "example refused");
    CHECK(ip.map[3] == 1 && ip.map[1] == -1 && ip.map[4] == -2 && ip.staged == std::vector<int32_t>({ 1, 4 }), "example without read past");
}

int main() {
    std::mt19937 rng(42);
    test_sched(rng);
    test_lookahead();
    test_copies(rng);
    test_inplace(rng);
    if (fails == 0) {
        printf("test-moe-ring: all OK\n");
    } else {
        printf("test-moe-ring: %d failures\n", fails);
    }
    return fails == 0 ? 0 : 1;
}
