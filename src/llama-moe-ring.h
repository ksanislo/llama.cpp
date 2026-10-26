#pragma once

// LLAMA_MOE_RING: bookkeeping of the expert ring of one device, no ggml calls (tests/test-moe-ring.cpp)
//
// A graph stages the experts of a device's layers in K layer-sized slots, slot = rank of the layer on its device % K (fixed by
// the graph). The ring streams every expert of the staged layers that is not in the cache on the device's copy stream, in layer
// order and without the routing, as far ahead as the slots allow: the copy into a slot waits for the compute stream to be done
// with the slot's previous layer. Partial layers (batches below the full-batch rule) need their routing, so they are staged at
// their own layer and the ring does not pass them.

#include <algorithm>
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

// LLAMA_MOE_RING_INPLACE: where ggml_mul_mat_id_slots reads each expert of a staged layer. A cached expert is read in place from
// its cache slot, an uncached one from its place in the staging slot (the ring copies only these), and beyond the staging slot's
// cap places from the host experts directly (mapped pinned memory). MMQ reads a little past the end of an expert when a row's
// length is not a multiple of its K step (Flash-Next's down projection, 640 wide); with read_past the bytes after a slot must be
// a valid expert of the same layout, so a cached expert is read in place only if the next slot holds an expert, and the host's
// last expert is never read from host memory (past it lies whatever follows the host tensor). The staging slot has one place
// more than cap, which takes zeros after the last staged expert.
struct llama_moe_ring_inplace {
    std::vector<int32_t> map;    // [n_expert] slot (>= 0), -1 - place (staged), host_code() (host experts)
    std::vector<int32_t> staged; // the staged experts by place, ascending
    std::vector<int32_t> host;   // the experts read from host memory (used ones)
    int32_t n_hit = 0;           // experts read in place from their cache slot (used ones)
    int32_t cap   = 0;

    int32_t host_code() const { return -1 - cap; }

    // slots[e] < n_slots: expert e is cached in slot slots[e]; key_of[s] >= 0: slot s holds an expert (only read with read_past);
    // used == nullptr: all experts are used, count (optional, with used): tokens per expert, the busiest uncached ones are staged
    // first; can_host: experts beyond cap may be read from host memory. Returns false if they would have to be (and !can_host).
    bool plan(const int32_t * slots, int32_t n_slots, const int32_t * key_of, int32_t n_expert, const std::vector<bool> * used,
            const std::vector<int32_t> * count, int32_t cap_, bool can_host, bool read_past) {
        cap = cap_;
        map.assign(n_expert, host_code());
        staged.clear();
        host.clear();
        n_hit = 0;

        std::vector<int32_t> miss;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (used != nullptr && !(*used)[e]) {
                continue; // never read: any value
            }
            const int32_t s = slots[e];
            const bool in_place = s >= 0 && s < n_slots && (!read_past || (s + 1 < n_slots && key_of[s + 1] >= 0));
            if (in_place) {
                map[e] = s;
                n_hit++;
            } else {
                miss.push_back(e);
            }
        }

        if ((int32_t) miss.size() <= cap) {
            staged = miss;
        } else {
            if (!can_host) {
                return false;
            }
            std::vector<int32_t> order = miss;
            if (count != nullptr) {
                std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return (*count)[a] > (*count)[b]; });
            }
            // the last expert stays out of host memory when MMQ reads past an expert
            const int32_t last = n_expert - 1;
            const bool keep_last = read_past && std::find(order.begin(), order.end(), last) != order.end();
            if (keep_last) {
                order.erase(std::find(order.begin(), order.end(), last));
                order.insert(order.begin(), last);
            }
            if (cap == 0 && keep_last) {
                return false;
            }
            staged.assign(order.begin(), order.begin() + cap);
            host.assign(order.begin() + cap, order.end());
            std::sort(staged.begin(), staged.end());
            std::sort(host.begin(), host.end());
        }
        for (size_t p = 0; p < staged.size(); ++p) {
            map[staged[p]] = -1 - (int32_t) p;
        }
        return true;
    }

    // LLAMA_MOE_RING_BYTES: the staging places from base on in a staging tensor of more places, host reads coded past all of them
    void place(int32_t base, int32_t host_places) {
        const int32_t code = host_code();
        for (int32_t & v : map) {
            if (v < 0) {
                v = v == code ? -1 - host_places : -1 - (base + (-1 - v));
            }
        }
    }
};

// LLAMA_MOE_RING_BYTES: the staging memory of a device as one byte ring per expert tensor position (gate, up, down), of which a
// staged layer takes exactly its uncached experts (+ one place of zeros), so the copy stream runs as many layers ahead as fit.
// A layer's experts sit at the same place index b in every ring (one map for its three matmuls): bytes [b*es[ip], (b+n)*es[ip]).
// A taken range is released when the matmuls that read it are queued (an event on the compute stream, release index r, event
// r % n_ev); a later range over released bytes waits for the newest such release first (the compute stream runs in order, so it
// covers the older ones). Ranges never overlap one that is not released.
struct llama_moe_ring_bytes {
    static constexpr int64_t n_ev = 64; // release events, reused round robin

    struct span {
        int64_t seq = -1;
        int64_t rel = -1;
        std::vector<size_t> off;
        std::vector<size_t> end;
    };

    std::vector<size_t> size;  // [ip] bytes of the ring
    std::vector<size_t> head;  // [ip] end of the last range taken
    std::vector<span>   live;  // taken, not released
    std::vector<span>   hist;  // released, the copy stream may not have waited for the release yet
    int64_t n_alloc = 0;
    int64_t n_rel   = 0;
    int64_t waited  = -1;      // the copy stream waited for every release <= waited

    void init(std::vector<size_t> sz) {
        size = std::move(sz);
        n_rel  = 0;
        waited = -1;
        reset();
    }

    // the copy stream waited for everything queued on the compute stream: all bytes are free
    void reset() {
        head.assign(size.size(), 0);
        live.clear();
        hist.clear();
        waited = n_rel - 1;
    }

    bool idle() const {
        return live.empty();
    }

    static bool overlaps(const span & a, const span & b) {
        for (size_t ip = 0; ip < a.off.size(); ++ip) {
            if (a.off[ip] < b.end[ip] && b.off[ip] < a.end[ip] && a.off[ip] < a.end[ip] && b.off[ip] < b.end[ip]) {
                return true;
            }
        }
        return false;
    }

    span make(int64_t b, int32_t n, const std::vector<size_t> & es) const {
        span s;
        for (size_t ip = 0; ip < size.size(); ++ip) {
            s.off.push_back((size_t) b*es[ip]);
            s.end.push_back((size_t) (b + n)*es[ip]);
        }
        return s;
    }

    // n places at place b fit in the ring (at most limit places) and miss every range not released
    bool fits(int64_t b, int32_t n, const std::vector<size_t> & es, int64_t limit) const {
        if (b < 0 || n <= 0 || b + n > limit) {
            return false;
        }
        const span s = make(b, n, es);
        for (size_t ip = 0; ip < size.size(); ++ip) {
            if (s.end[ip] > size[ip]) {
                return false;
            }
        }
        for (const span & l : live) {
            if (overlaps(s, l)) {
                return false;
            }
        }
        return true;
    }

    static int64_t place_at(const std::vector<size_t> & bytes, const std::vector<size_t> & es) {
        int64_t b = 0;
        for (size_t ip = 0; ip < bytes.size(); ++ip) {
            b = std::max(b, (int64_t) ((bytes[ip] + es[ip] - 1)/es[ip]));
        }
        return b;
    }

    // the place for n places of es[ip] bytes: after the last range, else from the start (wrap), else after a range not released;
    // -1 if none fits now
    int64_t find(const std::vector<size_t> & es, int32_t n, int64_t limit) const {
        const int64_t b = place_at(head, es);
        if (fits(b, n, es, limit)) {
            return b;
        }
        if (fits(0, n, es, limit)) {
            return 0;
        }
        for (const span & l : live) {
            const int64_t c = place_at(l.end, es);
            if (fits(c, n, es, limit)) {
                return c;
            }
        }
        return -1;
    }

    // take n places at b (find); returns the release the copy stream must wait for before writing them, -1 if none
    int64_t alloc(int64_t b, int32_t n, const std::vector<size_t> & es, int64_t & seq) {
        span s = make(b, n, es);
        s.seq = n_alloc++;
        int64_t w = -1;
        for (const span & h : hist) {
            if (overlaps(s, h)) {
                w = std::max(w, h.rel);
            }
        }
        if (w > waited) {
            waited = w;
            forget();
        } else {
            w = -1;
        }
        head = s.end;
        seq  = s.seq;
        live.push_back(std::move(s));
        return w;
    }

    struct released {
        int64_t rel   = -1; // record release event rel % n_ev on the compute stream now
        int64_t force = -1; // first: the copy stream waits for release event force % n_ev (it is about to be reused)
    };

    // the matmuls reading range seq are queued
    released release(int64_t seq) {
        released r;
        auto it = std::find_if(live.begin(), live.end(), [&](const span & s) { return s.seq == seq; });
        if (it == live.end()) {
            return r;
        }
        r.rel = n_rel++;
        for (const span & h : hist) {
            if (h.rel <= r.rel - n_ev && h.rel > waited) {
                r.force = std::max(r.force, h.rel);
            }
        }
        if (r.force >= 0) {
            waited = r.force;
            forget();
        }
        span s = std::move(*it);
        live.erase(it);
        s.rel = r.rel;
        hist.push_back(std::move(s));
        return r;
    }

    void forget() {
        hist.erase(std::remove_if(hist.begin(), hist.end(), [&](const span & h) { return h.rel <= waited; }), hist.end());
    }
};

// LLAMA_MOE_RING_BYTES: the plan of a graph on one device over the byte ring, as llama_moe_ring_sched over the slots: full layers
// are issued in order as far ahead as the ring has room (and at most max_ahead layers beyond the one computing); a layer is
// released at the next staged layer of the device or at the end of the graph
struct llama_moe_ring_bytes_sched {
    struct entry {
        int32_t il;
        bool    full;
        bool    issued = false;
        int32_t n      = 0;  // places taken (0: nothing staged)
        int64_t base   = -1; // first place
        int64_t seq    = -1; // the range in the ring
        int32_t ev     = -1; // ready event of its copies (llama_moe_cache)
        int32_t map    = -1; // pinned map (llama_moe_cache)
    };

    llama_moe_ring_bytes ring;
    std::vector<entry> plan;
    size_t  next_issue   = 0;
    size_t  next_consume = 0;
    int64_t consumed     = -1; // range of the layer read last, released at the next step
    int32_t max_ahead    = 32;

    void start(std::vector<entry> p, bool fence) {
        plan = std::move(p);
        next_issue   = 0;
        next_consume = 0;
        consumed     = -1;
        if (fence) {
            ring.reset();
        }
    }

    bool active() const {
        return next_consume < plan.size();
    }

    // the plan index to issue next ahead of its layer, -1 if none (the caller tries issue, which fails without room)
    int64_t can_issue() const {
        if (next_issue >= plan.size() || !plan[next_issue].full || next_issue - next_consume >= (size_t) max_ahead) {
            return -1;
        }
        return (int64_t) next_issue;
    }

    // entry i takes n places (es[ip] bytes each, at most limit places in its staging tensors): false if the ring has no room now;
    // wait = the release the copy stream waits for first, -1 if none
    bool issue(size_t i, int32_t n, const std::vector<size_t> & es, int64_t limit, int64_t & wait) {
        wait = -1;
        entry & e = plan[i];
        if (n > 0) {
            const int64_t b = ring.find(es, n, limit);
            if (b < 0) {
                return false;
            }
            wait   = ring.alloc(b, n, es, e.seq);
            e.base = b;
        }
        e.n      = n;
        e.issued = true;
        next_issue = i + 1;
        return true;
    }

    llama_moe_ring_bytes::released release() {
        llama_moe_ring_bytes::released r;
        if (consumed >= 0) {
            r = ring.release(consumed);
            consumed = -1;
        }
        return r;
    }

    int64_t at_layer(int32_t il) const {
        return next_consume < plan.size() && plan[next_consume].il == il ? (int64_t) next_consume : -1;
    }

    void consume(size_t i) {
        consumed     = plan[i].seq;
        next_consume = i + 1;
    }

    // layers issued ahead of the one computing
    size_t ahead() const {
        return next_issue > next_consume ? next_issue - next_consume : 0;
    }

    void drop() {
        plan.clear();
        next_issue   = 0;
        next_consume = 0;
        consumed     = -1;
    }
};
