// LLAMA_MOE_RING bookkeeping: the slot schedule (llama_moe_ring_sched) driven the way llama_moe_cache drives it, and the copy plan
// of a layer (llama_moe_ring_copies), against invariants:
// - a copy into a slot is queued only after the layer that read the slot before has been released (its matmuls queued) and, when
//   the copy is not behind a fence, waits for that release event
// - every staged layer finds its copies queued at its callback, in plan order, and a full layer is queued ahead when a slot allows
// - the copy plan covers every selected expert exactly once: uploaded if not cached, uploaded or copied from the cache if cached
// - LLAMA_MOE_RING_BYTES (llama_moe_ring_bytes_sched): random layer sizes over random ring sizes with wrap-around; a range is never
//   taken while a layer that may still read it is in flight, every copy over released bytes waits for (or is fenced past) the
//   release of their last reader, a waited release event still holds that release, and a layer reads the range it was given
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

// LLAMA_MOE_RING_BYTES: the byte ring driven like ring_begin_device / ring_copy_stage / ring_end_device, with per-byte owners
struct sim_bytes {
    llama_moe_ring_bytes_sched rb;
    int32_t n_pos;
    std::vector<std::vector<int64_t>> owner; // [ip][byte] range (seq) that last took the byte, -1 never
    std::vector<int64_t> rel_of;             // [seq] release index, -1 not released
    std::vector<int64_t> ev;                 // [n_ev] release index the event holds
    int64_t copy_waited = -1;                // the copy stream waited for every release <= this
    int64_t n_rel = 0;
    bool ring_prev = false, ring_fence = false;
    size_t ahead = 0, at_layer = 0, wraps = 0, waits = 0, forced = 0, max_ahead_seen = 0, layers = 0, ahead_sum = 0;

    sim_bytes(const std::vector<size_t> & size, int32_t max_ahead) : n_pos((int32_t) size.size()) {
        rb.ring.init(size);
        rb.max_ahead = max_ahead;
        for (size_t sz : size) {
            owner.emplace_back(sz, -1);
        }
        ev.assign(llama_moe_ring_bytes::n_ev, -1);
    }

    void wait(int64_t w) {
        CHECK(ev[w % llama_moe_ring_bytes::n_ev] == w, "wait for release %lld: its event was re-recorded (holds %lld)", (long long) w,
            (long long) ev[w % llama_moe_ring_bytes::n_ev]);
        copy_waited = std::max(copy_waited, w);
    }

    void record_release() {
        const auto r = rb.release();
        if (r.force >= 0) {
            wait(r.force);
            forced++;
        }
        if (r.rel >= 0) {
            CHECK(r.rel == n_rel, "release index %lld, expected %lld", (long long) r.rel, (long long) n_rel);
            ev[r.rel % llama_moe_ring_bytes::n_ev] = r.rel;
            n_rel++;
        }
        if (r.rel >= 0) {
            const int64_t seq = rb.ring.hist.back().seq;
            rel_of[seq] = r.rel;
        }
    }

    // try to issue entry i with n places of es; false without room
    bool fill(size_t i, int32_t n, const std::vector<size_t> & es, int64_t limit) {
        const size_t head0 = rb.ring.head[0];
        int64_t w = -1;
        if (!rb.issue(i, n, es, limit, w)) {
            return false;
        }
        const auto & e = rb.plan[i];
        if (w >= 0) {
            wait(w);
            waits++;
        }
        if (n == 0) {
            return true;
        }
        CHECK(e.base >= 0 && e.base + n <= limit, "range [%lld, %lld) past the limit %lld", (long long) e.base, (long long) (e.base + n), (long long) limit);
        if (e.base == 0 && head0 > 0) {
            wraps++;
        }
        if ((int64_t) rel_of.size() <= e.seq) {
            rel_of.resize(e.seq + 1, -1);
        }
        for (int32_t ip = 0; ip < n_pos; ++ip) {
            const size_t o = (size_t) e.base*es[ip], end = (size_t) (e.base + n)*es[ip];
            CHECK(end <= owner[ip].size(), "range past the ring");
            for (size_t b = o; b < end && b < owner[ip].size(); ++b) {
                const int64_t prev = owner[ip][b];
                if (prev >= 0) {
                    CHECK(rel_of[prev] != -1, "seq %lld overwrites bytes of seq %lld, which is not released", (long long) e.seq, (long long) prev);
                    CHECK(rel_of[prev] < 0 || rel_of[prev] <= copy_waited, "seq %lld overwrites bytes of seq %lld before waiting for its release %lld (waited %lld)",
                        (long long) e.seq, (long long) prev, (long long) rel_of[prev], (long long) copy_waited);
                }
                owner[ip][b] = e.seq;
            }
        }
        return true;
    }

    void begin(std::vector<llama_moe_ring_bytes_sched::entry> plan) {
        if (plan.empty()) {
            rb.drop();
            ring_prev = false;
            return;
        }
        const bool fence = !ring_prev || ring_fence || !rb.ring.idle();
        if (fence) {
            // the copy stream waits for the whole compute stream: ranges of a dropped plan were never read or are done
            copy_waited = n_rel - 1;
            for (const auto & s : rb.ring.live) {
                rel_of[s.seq] = -2;
            }
        }
        rb.start(std::move(plan), fence);
        ring_prev  = true;
        ring_fence = false;
    }

    template <typename F> void issue(F size_of) {
        for (int64_t i = rb.can_issue(); i >= 0; i = rb.can_issue()) {
            int32_t n; std::vector<size_t> es; int64_t limit;
            size_of((size_t) i, n, es, limit);
            if (!fill((size_t) i, n, es, limit)) {
                break;
            }
            ahead++;
            // the ready events and pinned maps of the layers in flight are reused round robin over n_ev
            CHECK(rb.ahead() <= (size_t) rb.max_ahead && rb.ahead() < (size_t) llama_moe_ring_bytes::n_ev,
                "%zu layers in flight, at most %d", rb.ahead(), rb.max_ahead);
        }
    }

    // the callback of layer il: false if the ring was dropped
    template <typename F> bool layer(int32_t il, F size_of) {
        if (!rb.active()) {
            return false;
        }
        const int64_t i = rb.at_layer(il);
        if (i < 0) {
            rb.drop();
            ring_fence = true;
            return false;
        }
        record_release();
        if (!rb.plan[i].issued) {
            int32_t n; std::vector<size_t> es; int64_t limit;
            size_of((size_t) i, n, es, limit);
            CHECK(rb.ring.idle(), "a layer not issued finds ranges in flight");
            const bool ok = fill((size_t) i, n, es, limit);
            CHECK(ok, "layer %d (%d places) does not fit an idle ring", il, n);
            at_layer++;
        }
        issue(size_of);
        const auto & e = rb.plan[i];
        // the layer reads its own range
        for (int32_t ip = 0; ip < n_pos && e.n > 0; ++ip) {
            int32_t n; std::vector<size_t> es; int64_t limit;
            size_of((size_t) i, n, es, limit);
            for (size_t b = (size_t) e.base*es[ip]; b < (size_t) (e.base + e.n)*es[ip]; ++b) {
                CHECK(owner[ip][b] == e.seq, "layer %d reads bytes of seq %lld instead of its own %lld", il, (long long) owner[ip][b], (long long) e.seq);
            }
        }
        rb.consume((size_t) i);
        max_ahead_seen = std::max(max_ahead_seen, rb.ahead());
        ahead_sum += rb.ahead();
        layers++;
        return true;
    }

    void end() {
        record_release();
    }
};

static void test_bytes(std::mt19937 & rng) {
    size_t n_wraps = 0, n_waits = 0, n_forced = 0, n_drops = 0, n_graphs = 0;
    for (int it = 0; it < 3000; ++it) {
        const int32_t n_pos   = 1 + rng() % 3;
        const int32_t n_group = 1 + rng() % 3;
        // per group the bytes of a place in each ring (Flash-Next: gate/up equal, down Q5_1 or Q8_0)
        std::vector<std::vector<size_t>> es(n_group, std::vector<size_t>(n_pos));
        for (auto & g : es) {
            for (auto & x : g) {
                x = 1 + rng() % 9;
            }
        }
        std::vector<int32_t> cap(n_group);
        std::vector<size_t> size(n_pos, 0);
        for (int32_t g = 0; g < n_group; ++g) {
            cap[g] = 1 + rng() % 20;
        }
        // at least two slots of the largest layer, sometimes much more
        const int32_t mult = 2 + (rng() % 4 == 0 ? rng() % 6 : 0);
        for (int32_t ip = 0; ip < n_pos; ++ip) {
            for (int32_t g = 0; g < n_group; ++g) {
                size[ip] = std::max(size[ip], (size_t) mult*(cap[g] + 1)*es[g][ip] + rng() % 7);
            }
        }
        std::vector<int64_t> limit(n_group);
        for (int32_t g = 0; g < n_group; ++g) {
            limit[g] = INT64_MAX;
            for (int32_t ip = 0; ip < n_pos; ++ip) {
                limit[g] = std::min<int64_t>(limit[g], (int64_t) (size[ip]/es[g][ip]));
            }
            // the staging tensors may hold fewer places than the bytes (the backend pads a tensor's allocation)
            limit[g] -= rng() % 2;
        }
        const int32_t n_layer = 1 + rng() % 14;
        std::vector<int32_t> group_of(n_layer);
        for (auto & g : group_of) {
            g = rng() % n_group;
        }
        sim_bytes d(size, 1 + rng() % 40);
        for (int graph = 0; graph < 8; ++graph) {
            n_graphs++;
            // places per layer in this graph: its misses (0 .. cap) + zeros, decided at issue (the cache does not change in a graph)
            std::vector<int32_t> places(n_layer);
            for (auto & p : places) {
                const int32_t g = group_of[&p - places.data()];
                const int32_t u = rng() % 5 == 0 ? 0 : (int32_t) (rng() % (cap[g] + 1));
                p = u == 0 ? 0 : u + 1;
            }
            std::vector<llama_moe_ring_bytes_sched::entry> plan;
            for (int32_t il = 0; il < n_layer; ++il) {
                if (rng() % 8 == 0) {
                    continue; // not staged in this graph
                }
                llama_moe_ring_bytes_sched::entry e;
                e.il   = il;
                e.full = rng() % 6 != 0;
                plan.push_back(e);
            }
            if (rng() % 10 == 0) {
                plan.clear(); // a graph without the ring
            }
            if (rng() % 7 == 0) {
                d.ring_fence = true; // fills, borrow, decode memory switch
            }
            auto size_of = [&](size_t i, int32_t & n, std::vector<size_t> & e_s, int64_t & lim) {
                const int32_t il = d.rb.plan[i].il;
                n   = places[il];
                e_s = es[group_of[il]];
                lim = limit[group_of[il]];
            };
            d.begin(plan);
            d.issue(size_of);
            for (size_t k = 0; k < plan.size(); ++k) {
                int32_t il = plan[k].il;
                if (rng() % 60 == 0) {
                    il = (il + 1) % n_layer; // not the planned layer: the ring is dropped
                }
                if (!d.layer(il, size_of)) {
                    n_drops++;
                    break;
                }
            }
            d.end();
        }
        n_wraps += d.wraps; n_waits += d.waits; n_forced += d.forced;
    }
    CHECK(n_wraps > 1000 && n_waits > 1000 && n_drops > 100, "coverage: %zu wraps, %zu waits, %zu forced, %zu drops in %zu graphs",
        n_wraps, n_waits, n_forced, n_drops, n_graphs);

    // event reuse: one small layer over a long stream of graphs without fences, ranges overlapped only rarely
    {
        sim_bytes d({ 1000 }, 32);
        for (int graph = 0; graph < 400; ++graph) {
            std::vector<llama_moe_ring_bytes_sched::entry> plan;
            for (int32_t il = 0; il < 3; ++il) {
                plan.push_back({ il, true });
            }
            auto size_of = [&](size_t i, int32_t & n, std::vector<size_t> & e_s, int64_t & lim) {
                n = i == 0 ? 2 : 1; e_s.assign(1, 1); lim = 1000;
            };
            d.begin(plan);
            d.issue(size_of);
            for (int32_t il = 0; il < 3; ++il) {
                d.layer(il, size_of);
            }
            d.end();
        }
        CHECK(d.forced > 0, "no release event was reused (forced %zu)", d.forced);
    }

    // Flash-Next-like: a ring of two slots of cap + 1 places, layers with about half the cap uncached: more than one layer ahead
    {
        const int32_t capf = 410;
        sim_bytes d({ (size_t) 2*(capf + 1)*3, (size_t) 2*(capf + 1)*3, (size_t) 2*(capf + 1)*4 }, 32);
        for (int graph = 0; graph < 6; ++graph) {
            std::vector<llama_moe_ring_bytes_sched::entry> plan;
            for (int32_t il = 0; il < 12; ++il) {
                plan.push_back({ il, true });
            }
            auto size_of = [&](size_t, int32_t & n, std::vector<size_t> & e_s, int64_t & lim) {
                n = 200 + 1; e_s.assign({ 3, 3, 4 }); lim = (int64_t) 2*(capf + 1);
            };
            d.begin(plan);
            d.issue(size_of);
            for (int32_t il = 0; il < 12; ++il) {
                d.layer(il, size_of);
            }
            d.end();
        }
        CHECK(d.max_ahead_seen >= 3 && d.at_layer == 0, "Flash-Next-like ring: at most %zu layers ahead, %zu at their layer", d.max_ahead_seen, d.at_layer);
        printf("test-moe-ring: bytes ring, 2 slots of 411 places, 200 misses a layer: %.2f layers ahead on average, max %zu\n",
            (double) d.ahead_sum/d.layers, d.max_ahead_seen);
    }

    // the map of an in-place plan placed in the ring
    for (int it = 0; it < 2000; ++it) {
        const int32_t n_expert = 2 + rng() % 40;
        const int32_t n_slots  = 1 + rng() % 40;
        std::vector<int32_t> slots(n_expert), key_of(n_slots, -1);
        for (int32_t e = 0; e < n_expert; ++e) {
            slots[e] = rng() % 2 ? (int32_t) (rng() % n_slots) : n_slots;
        }
        for (auto & k : key_of) {
            k = rng() % 3 ? 0 : -1;
        }
        const int32_t capi = rng() % (n_expert + 1);
        llama_moe_ring_inplace ip;
        if (!ip.plan(slots.data(), n_slots, key_of.data(), n_expert, nullptr, nullptr, capi, true, rng() % 2)) {
            continue;
        }
        const auto map0 = ip.map;
        const int32_t base = rng() % 100, hostp = base + capi + 1 + rng() % 50;
        ip.place(base, hostp);
        for (int32_t e = 0; e < n_expert; ++e) {
            if (map0[e] >= 0) {
                CHECK(ip.map[e] == map0[e], "placed map changed a cache slot");
            } else if (map0[e] == -1 - capi) {
                CHECK(ip.map[e] == -1 - hostp, "placed map: host read not past all places");
            } else {
                CHECK(ip.map[e] == -1 - (base + (-1 - map0[e])), "placed map: staging place not moved by base");
                CHECK(-1 - ip.map[e] < hostp, "placed map: staging place past the host code");
            }
        }
    }
    printf("test-moe-ring: bytes ring coverage %zu wraps, %zu waits, %zu forced, %zu drops in %zu graphs\n", n_wraps, n_waits, n_forced, n_drops, n_graphs);
}

int main() {
    std::mt19937 rng(42);
    test_sched(rng);
    test_lookahead();
    test_copies(rng);
    test_inplace(rng);
    test_bytes(rng);
    if (fails == 0) {
        printf("test-moe-ring: all OK\n");
    } else {
        printf("test-moe-ring: %d failures\n", fails);
    }
    return fails == 0 ? 0 : 1;
}
