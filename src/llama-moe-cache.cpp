#include "llama-moe-cache.h"
#include "llama-moe-lane.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-ring.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
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

// BEGIN moe_cache_policy (extracted by the CPU harness)

// whether slot s is in the SLRU lists (not a pending fill or a transient slot)
static bool moe_cache_slru_linked(const moe_cache_slru & c, int32_t s) {
    return c.prev[s] >= 0 || c.next[s] >= 0 || c.head[0] == s || c.head[1] == s;
}

// put the given slots in probation in this order (the first is the next victim), all others are unlinked
static void moe_cache_slru_relink(moe_cache_slru & c, const std::vector<int32_t> & order) {
    std::fill(c.prev.begin(), c.prev.end(), -1);
    std::fill(c.next.begin(), c.next.end(), -1);
    std::fill(c.is_protected.begin(), c.is_protected.end(), false);
    c.head[0] = c.tail[0] = c.head[1] = c.tail[1] = -1;
    c.n_protected = 0;
    for (int32_t s : order) {
        c.push_back(s);
    }
}

// expert usage profile: a score per (layer, expert), higher = used more
// own format, little endian: "LMCP", u32 version 1, n_layer, n_expert, f32 score[n_layer][n_expert]
// Strata's expert-profile.bin: "STRP", u32 version 1, n_layer, n_expert, slots, n_ranked, u16 (layer, expert)[n_ranked] hottest first
struct moe_cache_profile {
    int32_t n_layer  = 0;
    int32_t n_expert = 0;
    std::vector<float> score; // [n_layer*n_expert]
    bool strata = false;      // read from Strata's format (a rank, not counts)

    float get(int32_t il, int32_t e) const {
        return il < n_layer && e < n_expert ? score[(size_t) il*n_expert + e] : 0.0f;
    }
};

static bool moe_cache_profile_read(const std::string & path, int32_t n_expert, moe_cache_profile & p, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    char magic[4] = {};
    uint32_t hdr[5] = {};
    bool ok = fread(magic, 1, 4, f) == 4 && fread(hdr, 4, 3, f) == 3;
    if (ok && memcmp(magic, "STRP", 4) == 0) {
        ok = fread(hdr + 3, 4, 2, f) == 2;
    }
    if (!ok || (memcmp(magic, "LMCP", 4) != 0 && memcmp(magic, "STRP", 4) != 0) || hdr[0] != 1) {
        fclose(f);
        err = path + " is not a version 1 LMCP or STRP profile";
        return false;
    }
    if ((int32_t) hdr[2] != n_expert || hdr[1] == 0 || hdr[1] > 65536) {
        fclose(f);
        err = path + " is for " + std::to_string(hdr[2]) + " experts per layer, the model has " + std::to_string(n_expert);
        return false;
    }
    p.n_layer  = (int32_t) hdr[1];
    p.n_expert = n_expert;
    p.score.assign((size_t) p.n_layer*n_expert, 0.0f);
    p.strata = magic[0] == 'S';
    if (!p.strata) {
        ok = fread(p.score.data(), sizeof(float), p.score.size(), f) == p.score.size();
    } else {
        const uint32_t n_ranked = hdr[4];
        std::vector<uint16_t> pairs((size_t) n_ranked*2);
        ok = n_ranked <= (uint32_t) p.score.size() && fread(pairs.data(), 2, pairs.size(), f) == pairs.size();
        for (uint32_t r = 0; ok && r < n_ranked; ++r) {
            const int32_t il = pairs[2*r];
            const int32_t e  = pairs[2*r + 1];
            ok = il < p.n_layer && e < n_expert;
            float & s = p.score[(size_t) il*n_expert + e];
            if (ok && s == 0.0f) {
                s = (float) (n_ranked - r)/n_ranked;
            }
        }
    }
    fclose(f);
    if (!ok) {
        err = path + " is truncated or out of range";
    }
    return ok;
}

static bool moe_cache_profile_write(const std::string & path, int32_t n_layer, int32_t n_expert, const std::vector<float> & score, std::string & err) {
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "cannot create " + tmp;
        return false;
    }
    const uint32_t hdr[3] = { 1u, (uint32_t) n_layer, (uint32_t) n_expert };
    bool ok = fwrite("LMCP", 1, 4, f) == 4 && fwrite(hdr, 4, 3, f) == 3 && fwrite(score.data(), sizeof(float), score.size(), f) == score.size();
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        err = "cannot write " + path;
        return false;
    }
    return true;
}

// the hottest experts of the given layers with a score > 0, at most n, ordered by layer and expert
static std::vector<std::pair<int32_t, int32_t>> moe_cache_profile_pick(const moe_cache_profile & p, const std::vector<int32_t> & layers, int32_t n_expert, size_t n) {
    std::vector<std::pair<int32_t, int32_t>> res;
    for (int32_t il : layers) {
        for (int32_t e = 0; e < n_expert; ++e) {
            if (p.get(il, e) > 0.0f) {
                res.push_back({ il, e });
            }
        }
    }
    auto hotter = [&](const std::pair<int32_t, int32_t> & a, const std::pair<int32_t, int32_t> & b) {
        const float sa = p.get(a.first, a.second);
        const float sb = p.get(b.first, b.second);
        return sa != sb ? sa > sb : a < b;
    };
    if (res.size() > n) {
        std::nth_element(res.begin(), res.begin() + n, res.end(), hotter);
        res.resize(n);
    }
    std::sort(res.begin(), res.end());
    return res;
}

// LLAMA_MOE_CACHE_POLICY=adaptive: Strata's adaptive tier as a fill policy. Every expert has a decayed routing count
// (+1 per routing, x decay per graph); a missed expert enters the cache only if its count is at least min_count and beats the
// coldest cached expert's by gain (an empty slot takes any miss), at most fills_per_token per token
// counts are kept as log(u) + tick*k (k = -log(decay)), so decaying is free and the order of two experts is the order of their logs
struct moe_cache_policy {
    double  k         = 0.0;
    double  gain      = 1.5;
    double  min_count = 2.0;
    int64_t tick      = 0;
    std::vector<double> lg; // [n_layer*n_expert]

    void init(size_t n_keys, double decay, double gain, double min_count) {
        this->k         = -std::log(std::clamp(decay, 1e-6, 1.0));
        this->gain      = gain;
        this->min_count = min_count;
        lg.assign(n_keys, -INFINITY);
    }

    double value(int32_t key) const {
        return std::isinf(lg[key]) ? 0.0 : std::exp(lg[key] - (double) tick*k);
    }

    void add(int32_t key, double count) {
        lg[key] = std::log(value(key) + count) + (double) tick*k;
    }

    struct swap {
        double  gain;
        int32_t il;
        int32_t expert;
        int32_t slot; // empty or the victim's
    };

    // pair the candidates (misses of the group's layers, not cached) with free slots and the coldest cached experts,
    // hottest candidate with the coldest victim, until a candidate does not beat its victim by gain
    // with excl_layer >= 0, the experts of that layer marked seen in c are not victims (the graph reads them)
    void pair(const moe_cache_slru & c, std::vector<std::pair<int32_t, int32_t>> & cands, int32_t excl_layer, std::vector<swap> & out,
            std::vector<std::pair<double, int32_t>> & victims) const {
        if (cands.empty()) {
            return;
        }
        auto key = [&](const std::pair<int32_t, int32_t> & ie) { return ie.first*c.n_expert + ie.second; };
        std::sort(cands.begin(), cands.end(), [&](const auto & a, const auto & b) {
            return lg[key(a)] != lg[key(b)] ? lg[key(a)] > lg[key(b)] : a < b;
        });
        // cheap exit: no candidate could evict and the next victim of the lists is no free slot
        const bool maybe_free = c.head[0] >= 0 && c.key_of[c.head[0]] < 0;
        if (value(key(cands[0])) < min_count && !maybe_free) {
            return;
        }
        victims.clear();
        std::vector<int32_t> free_slots;
        for (int32_t s = 0; s < c.n_slots; ++s) {
            if (!moe_cache_slru_linked(c, s)) {
                continue;
            }
            const int32_t kv = c.key_of[s];
            if (kv < 0) {
                if (free_slots.size() < cands.size()) {
                    free_slots.push_back(s);
                }
                continue;
            }
            if (excl_layer >= 0 && kv / c.n_expert == excl_layer && c.seen[kv % c.n_expert] == c.seen_gen) {
                continue;
            }
            victims.push_back({ lg[kv], s });
        }
        const size_t n_v = std::min(victims.size(), cands.size() - std::min(cands.size(), free_slots.size()));
        std::partial_sort(victims.begin(), victims.begin() + n_v, victims.end());
        size_t iv = 0;
        for (size_t i = 0; i < cands.size(); ++i) {
            const double u = value(key(cands[i]));
            if (i < free_slots.size()) {
                out.push_back({ u, cands[i].first, cands[i].second, free_slots[i] });
                continue;
            }
            if (iv >= n_v) {
                break;
            }
            const double v = std::isinf(victims[iv].first) ? 0.0 : std::exp(victims[iv].first - (double) tick*k);
            if (u < min_count || u < v + gain) {
                break;
            }
            out.push_back({ u - v, cands[i].first, cands[i].second, victims[iv].second });
            iv++;
        }
    }
};

// slots [0, n) of a group are out of the cache: the classic path uploads the misses the policy does not admit there,
// mapped until the group's next plan or the next graph
struct moe_cache_transient {
    int32_t n    = 0;
    int32_t next = 0;
    std::vector<std::pair<int32_t, int32_t>> used; // (layer, expert)
};

static void moe_cache_transient_clear(moe_cache_slru & c, moe_cache_transient & tr) {
    for (const auto & ie : tr.used) {
        int32_t & s = c.slot_map[ie.first][ie.second];
        if (s < tr.n) {
            s = c.empty;
        }
    }
    tr.used.clear();
    tr.next = 0;
}

struct moe_cache_policy_stats {
    size_t admitted  = 0; // fills chosen by the policy
    size_t into_free = 0; // of these, into empty slots
    size_t kept_out  = 0; // misses not admitted
    size_t transient = 0; // of these, read from a transient slot (classic path)
    size_t forced    = 0; // of these, cached anyway: no transient slot left (classic path)
    size_t capped    = 0; // admissions over the per-token cap
};

struct moe_cache_policy_scratch {
    std::vector<moe_cache_policy::swap> swaps;
    std::vector<std::pair<int32_t, int32_t>> cands;
    std::vector<std::pair<double, int32_t>> victims;
};

// the classic path's plan with the adaptive policy: hits are touched, a miss is cached if the policy admits it (into an empty
// slot always, over a victim while budget = evictions left this token > 0), otherwise uploaded to a transient slot, or, with none left, cached into the SLRU victim (forced)
// returns false if the ids select more distinct experts than the cache has slots
static bool moe_cache_plan_adaptive(moe_cache_slru & c, moe_cache_transient & tr, const moe_cache_policy & ad, int64_t & budget,
        int32_t il, const int32_t * ids, size_t n_ids, std::vector<moe_cache_slru::fill> & fills, size_t & n_hit,
        moe_cache_policy_stats & st, moe_cache_policy_scratch & sc) {
    moe_cache_transient_clear(c, tr);
    fills.clear();
    n_hit = 0;

    c.mark_seen(ids, n_ids);
    if (c.uniq.size() > (size_t) (c.n_slots - tr.n)) {
        return false;
    }
    int32_t * slots = c.slot_map[il];
    c.misses.clear();
    for (int32_t e : c.uniq) {
        if (slots[e] < c.n_slots) {
            c.touch(slots[e]);
            n_hit++;
        } else {
            c.misses.push_back(e);
        }
    }
    if (c.misses.empty()) {
        return true;
    }

    sc.cands.clear();
    for (int32_t e : c.misses) {
        sc.cands.push_back({ il, e });
    }
    sc.swaps.clear();
    ad.pair(c, sc.cands, il, sc.swaps, sc.victims);
    // the miss is uploaded anyway: an empty slot costs nothing, only evictions count against the budget
    size_t n_free = 0;
    while (n_free < sc.swaps.size() && c.key_of[sc.swaps[n_free].slot] < 0) {
        n_free++;
    }
    const size_t n_adm = n_free + std::min<size_t>(sc.swaps.size() - n_free, (size_t) std::max<int64_t>(0, budget));
    st.capped += sc.swaps.size() - n_adm;
    budget -= (int64_t) (n_adm - n_free);

    auto cache = [&](int32_t e, int32_t s) {
        if (c.key_of[s] >= 0) {
            c.slot_map[c.key_of[s] / c.n_expert][c.key_of[s] % c.n_expert] = c.empty;
        } else {
            c.n_filled++;
        }
        c.key_of[s] = il*c.n_expert + e;
        slots[e] = s;
        c.move(s, false);
        fills.push_back({ e, s });
    };
    for (size_t i = 0; i < n_adm; ++i) {
        st.into_free += c.key_of[sc.swaps[i].slot] < 0;
        cache(sc.swaps[i].expert, sc.swaps[i].slot);
    }
    st.admitted += n_adm;

    for (int32_t e : c.misses) {
        if (slots[e] < c.n_slots) {
            continue; // admitted
        }
        st.kept_out++;
        if (tr.next < tr.n) {
            const int32_t s = tr.next++;
            slots[e] = s;
            tr.used.push_back({ il, e });
            fills.push_back({ e, s });
            st.transient++;
            continue;
        }
        cache(e, c.victim(il));
        st.forced++;
    }
    // sorted by expert, consecutive slots are uploaded together
    std::sort(fills.begin(), fills.end(), [](const moe_cache_slru::fill & x, const moe_cache_slru::fill & y) { return x.expert < y.expert; });
    return true;
}

// END moe_cache_policy

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

// BEGIN moe_cache_plan (extracted by the CPU harness)
// slots a group keeps below its lent tail and the staging slots borrowed under it: the cache path of the largest batch it
// serves (up to n_used_batch distinct experts, at most a layer's) and its transient slots (adaptive policy, else 0)
static int32_t moe_cache_keep_slots(int32_t n_expert, int64_t n_used_batch, int32_t n_transient) {
    return (int32_t) std::min<int64_t>(n_expert, n_used_batch) + n_transient;
}

// the slots a group needs so that batches of n_used_batch selected experts take the cache path (use_cache) beside the transient slots
static int32_t moe_cache_floor_slots(int32_t n_expert, int64_t n_used_batch, int32_t max_slots) {
    const int32_t need = (int32_t) std::min<int64_t>(n_expert, n_used_batch);
    return std::min(max_slots, need + need/7 + 1);
}

// bank slots under n_stage staging slots of sec[ip] bytes in bank ip (nb[ip] bytes per slot)
static int32_t moe_cache_stage_slots(const std::vector<size_t> & nb, const std::vector<size_t> & sec, int32_t n_stage) {
    int32_t k = 1;
    for (size_t ip = 0; ip < sec.size() && ip < nb.size(); ++ip) {
        k = std::max(k, (int32_t) ((n_stage*sec[ip] + nb[ip] - 1)/nb[ip]));
    }
    return k;
}

// raise the budgets below their floors to the floors, taken from the budgets above theirs in proportion to the excess;
// nothing changes if the budgets above cannot pay for it
static bool moe_cache_plan_floors(std::vector<double> & budget, const std::vector<double> & floor) {
    double lack  = 0.0;
    double above = 0.0;
    for (size_t i = 0; i < budget.size(); ++i) {
        if (budget[i] < floor[i]) {
            lack += floor[i] - budget[i];
        } else {
            above += budget[i] - floor[i];
        }
    }
    if (lack <= 0.0 || above < lack) {
        return false;
    }
    for (size_t i = 0; i < budget.size(); ++i) {
        budget[i] = budget[i] < floor[i] ? floor[i] : budget[i] - (budget[i] - floor[i])*lack/above;
    }
    return true;
}

// LLAMA_MOE_DECODE_MEM: one device's groups in a VMM buffer, each bank on a granule; the tail [first, n_slots) of a group is
// lent to the compute buffer of large batches in whole granules. Each group is asked for `lend` bytes and keeps `min_first`
// slots (moe_cache_keep_slots, plus the staging slots of LLAMA_MOE_RING in the group that holds them); what a group cannot
// lend goes to the groups with room left, largest room first.
struct moe_cache_plan_group {
    std::vector<size_t> nb;   // [bank] bytes per slot
    std::vector<size_t> size; // [bank] allocation size of the bank
    int32_t n_slots   = 0;
    int32_t min_first = 0;
    double  lend      = 0.0;
    // result
    std::vector<size_t> off;  // [bank] offset in the buffer
    int32_t first = 0;        // n_slots if nothing is lent
    std::vector<std::pair<size_t, size_t>> ranges; // (offset, size) released while lent
};

// returns the buffer size, `lent` the bytes released while lent
static size_t moe_cache_plan_lend(std::vector<moe_cache_plan_group> & gs, size_t G, size_t & lent) {
    size_t cur = 0;
    for (auto & g : gs) {
        g.off.clear();
        g.ranges.clear();
        g.first = g.n_slots;
        for (size_t ip = 0; ip < g.size.size(); ++ip) {
            const size_t off = GGML_PAD(cur, G);
            g.off.push_back(off);
            cur = off + g.size[ip];
        }
    }
    const size_t total = GGML_PAD(std::max<size_t>(cur, 1), G);

    auto released = [G](const moe_cache_plan_group & g, int32_t first) {
        size_t res = 0;
        for (size_t ip = 0; ip < g.nb.size(); ++ip) {
            const size_t lo = GGML_PAD(g.off[ip] + (size_t) first*g.nb[ip], G);
            const size_t hi = (g.off[ip] + (size_t) g.n_slots*g.nb[ip])/G*G;
            res += hi > lo ? hi - lo : 0;
        }
        return res;
    };
    std::vector<double> want(gs.size(), 0.0);
    std::vector<double> room(gs.size(), 0.0);
    double short_sum = 0.0;
    for (size_t i = 0; i < gs.size(); ++i) {
        const auto & g = gs[i];
        if (g.n_slots == 0) {
            short_sum += g.lend; // a group without slots passes its share on
            continue;
        }
        const double cap = g.n_slots > g.min_first ? (double) released(g, g.min_first) : 0.0;
        want[i]    = std::min(g.lend, cap);
        room[i]    = cap - want[i];
        short_sum += g.lend - want[i];
    }
    while (short_sum > 0.0) {
        int32_t best = -1;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (room[i] > 0.0 && (best < 0 || room[i] > room[best])) {
                best = (int32_t) i;
            }
        }
        if (best < 0) {
            break;
        }
        const double add = std::min(short_sum, room[best]);
        want[best] += add;
        room[best] -= add;
        short_sum  -= add;
    }

    lent = 0;
    for (size_t i = 0; i < gs.size(); ++i) {
        auto & g = gs[i];
        if (g.n_slots == 0 || want[i] <= 0.0) {
            continue;
        }
        size_t per_slot = 0;
        for (size_t nb : g.nb) {
            per_slot += nb;
        }
        int32_t first = std::max(g.min_first, g.n_slots - (int32_t) (want[i]/(double) per_slot));
        while (first > g.min_first && (double) released(g, first) < want[i]) {
            first--;
        }
        if (first >= g.n_slots || released(g, first) == 0) {
            continue;
        }
        g.first = first;
        for (size_t ip = 0; ip < g.nb.size(); ++ip) {
            const size_t lo = GGML_PAD(g.off[ip] + (size_t) first*g.nb[ip], G);
            const size_t hi = (g.off[ip] + (size_t) g.n_slots*g.nb[ip])/G*G;
            if (hi > lo) {
                g.ranges.push_back({ lo, hi - lo });
                lent += hi - lo;
            }
        }
    }
    return total;
}

// device memory at its peak: during a large batch (the lent tail backs the compute buffer and stage_lent, the ring's staging
// slots, the rest of them is new memory) or between them (the whole cache and the small compute buffer); without decode
// memory the compute buffer is always large
static size_t moe_cache_plan_peak(size_t cache, size_t lent, size_t stage_own, size_t stage_lent, size_t large, size_t small, bool lends) {
    if (!lends) {
        return cache + stage_own + large;
    }
    return std::max(cache - std::min(cache, lent) + stage_own + stage_lent + large, cache + stage_own + small);
}
// the cache large batches use (prompt-time): each group's slots below its lent tail (first = n_slots if it lends none), less
// the staging slots the ring borrows from the lender without decode memory (with it they are in the lent memory)
static size_t moe_cache_plan_prompt_bytes(const std::vector<size_t> & per_slot, const std::vector<int32_t> & first, int32_t lender, int32_t stage_k) {
    size_t res = 0;
    for (size_t i = 0; i < per_slot.size(); ++i) {
        const int32_t n = first[i] - ((int32_t) i == lender ? std::min(first[i], stage_k) : 0);
        res += per_slot[i]*(size_t) std::max(0, n);
    }
    return res;
}

// LLAMA_MOE_DECODE_MEM: small batches run on the whole cache, so the floors apply to budget + lend; a total above all of a
// group's experts (cap) goes to the groups with room (else the lend asked of a small group is lost and large batches keep
// less cache than without decode memory); the lend is then spread again in proportion to the new totals (the sums stay)
static bool moe_cache_plan_floors_dm(std::vector<double> & budget, std::vector<double> & lend, const std::vector<double> & floor,
        const std::vector<double> & cap) {
    std::vector<double> tot(budget.size());
    double sum_t = 0.0;
    double sum_l = 0.0;
    for (size_t i = 0; i < budget.size(); ++i) {
        tot[i] = budget[i] + lend[i];
        sum_t += tot[i];
        sum_l += lend[i];
    }
    if (sum_t <= 0.0) {
        return false;
    }
    bool changed = budget.size() > 1 && moe_cache_plan_floors(tot, floor);
    for (int round = 0; round < 16; ++round) {
        double over = 0.0;
        double room = 0.0;
        for (size_t i = 0; i < tot.size(); ++i) {
            over += std::max(0.0, tot[i] - cap[i]);
            room += std::max(0.0, cap[i] - tot[i]);
        }
        if (over < 1.0 || room < 1.0) {
            break;
        }
        const double give = std::min(over, room);
        for (size_t i = 0; i < tot.size(); ++i) {
            if (tot[i] > cap[i]) {
                tot[i] -= (tot[i] - cap[i])*give/over;
            } else {
                tot[i] += (cap[i] - tot[i])*give/room;
            }
        }
        changed = true;
    }
    if (!changed) {
        return false;
    }
    for (size_t i = 0; i < budget.size(); ++i) {
        lend[i]   = tot[i]*sum_l/sum_t;
        budget[i] = tot[i] - lend[i];
    }
    return true;
}

// LLAMA_MOE_CACHE_SPLIT=stream: per prompt chunk each stage streams need - budget (need = its host experts + the ring's staging
// borrowed from its cache) and the stage that streams most paces the pipe. Water-filling: the budgets in [0, min(cap, need)],
// summing to at most size, that minimize the largest need - budget. Returns that largest need - budget.
static double moe_cache_split_stream(const std::vector<double> & need, const std::vector<double> & cap, double size, std::vector<double> & budget) {
    const size_t n = need.size();
    std::vector<double> hi(n);
    double sum_hi = 0.0;
    double top    = 0.0;
    for (size_t i = 0; i < n; ++i) {
        hi[i]   = std::max(0.0, std::min(cap[i], need[i]));
        sum_hi += hi[i];
        top     = std::max(top, need[i]);
    }
    auto at = [&](double level, std::vector<double> & b) {
        double sum = 0.0;
        b.resize(n);
        for (size_t i = 0; i < n; ++i) {
            b[i] = std::clamp(need[i] - level, 0.0, hi[i]);
            sum += b[i];
        }
        return sum;
    };
    if (size >= sum_hi) {
        at(0.0, budget);
    } else {
        // the sum falls as the level rises: the lowest level whose budgets fit in size
        double lo = 0.0;
        double up = top;
        std::vector<double> tmp;
        for (int it = 0; it < 64 && up - lo > 1.0; ++it) {
            const double mid = 0.5*(lo + up);
            if (at(mid, tmp) > size) {
                lo = mid;
            } else {
                up = mid;
            }
        }
        at(up, budget);
    }
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        worst = std::max(worst, need[i] - budget[i]);
    }
    return worst;
}
// END moe_cache_plan

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
        size_t ib = 0;         // index in the backends of the context
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
        int32_t borrow_end   = 0;                     // slots [borrow_first, borrow_end); below the lent tail with LLAMA_MOE_DECODE_MEM
        bool borrowed = false;                        // the cache slots under the staging slots are out of the SLRU

        // memory plan (plan_memory): the group the staging slots borrow from and its slots under them, the cache buffer, the
        // bytes lent to large batches, the staging bytes of their own (not borrowed); fit(): budget cut to fit the device
        int32_t plan_lender    = -1;
        int32_t plan_stage_k   = 0;
        size_t  plan_total     = 0;
        size_t  plan_lent      = 0;
        size_t  plan_stage_own = 0;
        double  fit_cut        = 0.0;
        double  fit_last_cut   = 0.0;
        std::vector<std::pair<size_t, double>> fit_saved; // (group, budget) before the last cut
        double  fit_excess     = 0.0;
        bool    fit_stuck      = false;
        size_t  pt_nodm        = 0; // the cache large batches use without LLAMA_MOE_DECODE_MEM (moe_cache_plan_prompt_bytes)

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

        // LLAMA_MOE_RING: the staging slots are a ring the copy stream fills ahead of the compute stream (see ring_begin)
        llama_moe_ring_sched rs;
        std::vector<ggml_backend_event_ptr> ev_used;    // [n_stage] the compute stream is done with the slot's last layer
        std::unique_ptr<std::mutex> mtx;                // the ring state, if the scheduler calls back from a thread per device
        ggml_backend_buffer_ptr buf_pinned;             // identity map and zeros, pinned: a copy from them never waits on the host
        const int32_t * pinned_map   = nullptr;
        const uint8_t * pinned_zeros = nullptr;
        size_t pinned_map_size = 0;
        bool ring_prev  = false;                        // the last graph used the ring on the device
        bool ring_fence = false;                        // since then the compute stream changed the banks or the slots another way
        llama_moe_ring_copies r_cp;
        std::vector<std::vector<llama_moe_ring_sched::entry>> ring_plans; // [copy] submit threads: the plan of the graph in that copy
        std::vector<void *> r_dst;
        std::vector<const void *> r_src;
        std::vector<int32_t> r_ids;
        std::vector<bool> r_used;
        struct ring_stats {
            size_t graphs       = 0; // graphs with a plan on the device
            size_t layers       = 0; // streamed ahead of their layer
            size_t at_layer     = 0; // full layers issued only at their own layer (no free slot before)
            size_t partial      = 0; // only the selected experts, after reading them back
            size_t dropped      = 0; // graphs whose layers did not follow the plan (the per-layer staging took over)
            size_t copies       = 0; // host -> device copies queued
            size_t launches     = 0; // copy kernels for the cached experts
            size_t bytes_host   = 0;
            size_t bytes_cache  = 0;
            int64_t issue_us    = 0; // host time queueing the copies
            int64_t issue_us_max = 0;
        } rstats;
        ring_stats rstats_graph; // LLAMA_MOE_RING_LOG: rstats at the start of the device's part of the graph
        int64_t    rgraph_t0 = 0;
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
        moe_cache_transient tr;                        // LLAMA_MOE_CACHE_POLICY=adaptive

        // LLAMA_MOE_DECODE_MEM: slots [lend_first, n_slots) are out of the cache while the device lends their memory
        double  budget     = 0.0;
        double  lend       = 0.0;                      // bytes this group lends
        int32_t lend_first = 0;
        std::vector<std::pair<size_t, size_t>> lend_ranges; // (offset, size) in the device buffer, whole granules
        std::vector<size_t> bank_offs;                 // offset of each bank in the device buffer
        bool tail_out = false;                         // the tail stayed out after a give (no memory to back it), see dm_switch
        std::vector<std::pair<int32_t, int32_t>> refill; // (layer, expert) hottest first, uploaded into the returned slots
        size_t refill_pos = 0;
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
        std::vector<const uint8_t *> ring_host; // LLAMA_MOE_RING: device address of the host experts (mapped pinned memory), per bank

        // LLAMA_MOE_CPU_LANE
        uint8_t * lane_host = nullptr;       // the channel of the layer in pinned host memory
        void    * lane_dev  = nullptr;       // its address on the device
        bool      lane_warned = false;
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
    int32_t ring       = 0; // LLAMA_MOE_RING=K: the K staging slots are filled as a ring, all uncached experts of a device's layers
    int32_t ring_merge = 0; // LLAMA_MOE_RING_MERGE=g: up to g cached experts between two uploaded runs are uploaded with them
    int32_t ring_zc     = 1;  // LLAMA_MOE_RING_ZC=0: uncached experts by host copies on the copy engine (one per run) instead of the kernel
    int32_t ring_blocks = 16; // LLAMA_MOE_RING_BLOCKS: CUDA blocks of the copy kernel
    bool    ring_prio   = false; // LLAMA_MOE_RING_PRIO=1: the copy stream has the device's highest priority (its kernel's blocks go first)
    bool    ring_log    = false; // LLAMA_MOE_RING_LOG=1: one line per device and ring graph (the exit stats are often lost with the log)
    typedef void (*copy_blocks_t)(ggml_backend_t, void * const *, const void * const *, int, size_t, int);
    copy_blocks_t copy_blocks = nullptr;
    std::unordered_map<const ggml_tensor *, int32_t> stage_view_layer; // staged experts (the matmul's src[0]) -> layer
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

    // LLAMA_MOE_CPU_LANE=f (needs LLAMA_MOE_DEVICE_HITS): of the misses of a device-path layer, the first round(f * misses) in
    // routing order are computed by the CPU (llama_moe_lane, a thread pool of its own) while the device computes the hits and
    // reads the other misses over PCIe. The graph meets the CPU through a channel per layer in pinned host memory mapped to
    // the device: ggml_moe_doorbell (after the slot lookup) publishes x and the plan and rings, ggml_moe_join (after the down
    // projection) spins on the done word and copies the CPU's rows in. No host step, no stream sync, CUDA graphs stay on.
    float   lane_frac       = 0.0f;
    int32_t lane_timeout_ms = 2000;
    std::vector<ggml_backend_buffer_ptr> lane_bufs;   // the channels of each device, pinned host memory
    std::vector<ggml_backend_buffer_ptr> lane_mapped; // the same mapped to the device
    std::unique_ptr<llama_moe_lane> lane;             // after the channels: stopped before they are freed

    // --moe-cache-profile / LLAMA_MOE_CACHE_PROFILE: each device's cache starts with the hottest experts of its layers; the
    // routings of small batches are counted and saved back (blended with the loaded profile) at exit and periodically
    std::string prof_path;
    std::string prof_save;            // empty: not saved
    moe_cache_profile prof;           // n_layer 0 if none
    std::vector<double> prof_run;     // [n_layer*n_expert] routings of this run
    double  prof_run_sum    = 0.0;
    double  prof_saved_sum  = 0.0;
    double  prof_keep       = 0.5;    // weight of the loaded profile in the saved one
    double  prof_save_min   = 10.0;   // minutes between periodic saves, 0 = only at exit
    int64_t prof_saved_us   = 0;
    int64_t prof_graphs     = 0;
    int32_t n_expert_model  = 0;

    // LLAMA_MOE_CACHE_POLICY=adaptive (see moe_cache_policy), for the classic path and the device-hits fills
    bool    ad_on = false;
    moe_cache_policy ad;
    int32_t ad_fills_per_token = 16;
    int64_t ad_budget      = 0;
    int64_t ad_budget_tick = -1;
    moe_cache_policy_stats stats_ad;
    moe_cache_policy_scratch ad_scratch;
    std::vector<moe_cache_policy::swap> ad_swaps;
    std::vector<moe_cache_policy::swap> ad_swaps_g;
    std::vector<std::pair<double, int32_t>> ad_victims;
    std::vector<std::vector<std::pair<int32_t, int32_t>>> ad_cands_g; // [group]

    // LLAMA_MOE_DECODE_MEM=1: the compute buffer of a device is VMM address space whose part beyond `small` is backed only while
    // a graph uses it (large batches); meanwhile the same memory is released from the tails of the device's banks, whose
    // coldest experts leave the cache (the hotter ones in the tails move down). After the batch the compute memory is released,
    // the tails are backed again and refilled with the hottest experts missing (adaptive counts, then the profile)
    struct dm_device {
        size_t lend  = 0;     // bytes the banks give up for large batches, 0 if the device does not lend
        size_t stage = 0;     // of them, LLAMA_MOE_RING's staging slots: address space backed only while lent (init_stage)
        size_t large = 0;     // measured compute buffer of the worst-case batch
        size_t small = 0;     // compute bytes backed for small batches
        size_t gran  = 0;
        ggml_backend_buffer_type_t cbuft = nullptr; // compute buffer type (VMM address space)
        ggml_backend_buffer_t cbuf = nullptr;       // the scheduler's compute buffer last seen
        size_t committed = 0;                        // backed prefix of cbuf
        bool   lent      = false;                    // the bank tails are released
        int32_t n_small  = 0;                        // small graphs in a row while lent
        int32_t n_retry  = 0;                        // small graphs while not lent (retry of the tails that stayed out)
        size_t used_small_max = 0;
        size_t used_large_max = 0;
        // LLAMA_MOE_DECODE_MEM_ALIAS: the lent tails are mapped into the staging slots and the compute buffer for good, so a
        // switch only evicts; tail space = the device's lent ranges in group order: [0, alias_stage) the staging slots, then
        // [small, small + alias_comp) of cbuf
        std::vector<std::pair<size_t, size_t>> tails; // (offset in the cache buffer, size)
        size_t alias_stage = 0;
        size_t alias_comp  = 0;
    };
    struct dm_cost { // microseconds, all switches
        int64_t sync = 0, evict = 0, tail = 0, comp = 0, stage = 0, rest = 0;
    };
    struct dm_stats {
        size_t  takes    = 0;
        size_t  gives    = 0;
        int64_t take_us  = 0;
        int64_t give_us  = 0;
        size_t  moved    = 0;
        size_t  evicted  = 0;
        size_t  released = 0; // bytes released from the banks, all takes
        size_t  refilled = 0;
        size_t  refill_bytes  = 0;
        size_t  refill_graphs = 0;
        size_t  tails_out = 0;   // group tails a give could not back
        dm_cost take_cost;
        dm_cost give_cost;
    };
    typedef ggml_backend_buffer_t      (*vmm_alloc_t)(ggml_backend_buffer_type_t, size_t);
    typedef bool                       (*vmm_commit_t)(ggml_backend_buffer_t, size_t, size_t);
    typedef size_t                     (*vmm_release_t)(ggml_backend_buffer_t, size_t, size_t);
    typedef size_t                     (*vmm_gran_t)(ggml_backend_buffer_type_t);
    typedef ggml_backend_buffer_type_t (*vmm_cbuft_t)(ggml_backend_dev_t);
    bool    dm_on           = false;
    int32_t dm_return_after = 1;   // small graphs in a row before the tails come back
    int32_t dm_refill_cap   = 256; // refills per device per graph, 0 = all at once
    float   dm_compact      = 16.0f; // re-plan a small graph laid out by a large one (not one of a similar size: MTP windows)
    vmm_alloc_t   vmm_alloc   = nullptr;
    vmm_commit_t  vmm_commit  = nullptr;
    vmm_release_t vmm_release = nullptr;
    // spare granules: the memory moves between the banks, the compute buffer and the staging slots without cuMemCreate/Release
    typedef size_t (*vmm_release_spare_t)(ggml_backend_buffer_t, size_t, size_t);
    typedef bool   (*vmm_commit_spare_t)(ggml_backend_buffer_t, size_t, size_t, bool);
    typedef size_t (*vmm_trim_t)(ggml_backend_buffer_type_t);
    vmm_release_spare_t vmm_release_spare = nullptr;
    vmm_commit_spare_t  vmm_commit_spare  = nullptr;
    vmm_trim_t          vmm_trim          = nullptr;
    typedef bool (*vmm_alias_t)(ggml_backend_buffer_t, size_t, ggml_backend_buffer_t, size_t, size_t);
    vmm_commit_t vmm_chunk    = nullptr;
    vmm_alias_t  vmm_alias_fn = nullptr;
    bool         dm_alias     = false; // LLAMA_MOE_DECODE_MEM_ALIAS (default 1 where the backend can)
    std::vector<dm_device> dm_devs;
    dm_stats stats_dm;
    bool dm_warned = false;
    mutable bool graph_stages = false; // the graph built last reads staging slots (get_slot_map), read and reset by on_alloc

    // LLAMA_MOE_CACHE_SPLIT=miss: the hottest profiled experts are cached first, on whichever device they are
    bool split_miss  = false;
    // LLAMA_MOE_CACHE_SPLIT=stream: the budget evens out the experts each stage streams per prompt chunk
    bool split_stream = false;
    bool prof_loaded = false;
    bool prof_exists = false;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;
    // per device for copy_experts: with submit threads each device's copies are queued by its own thread
    std::vector<ggml_context_ptr> ctx_views_dev;

    // with submit threads: the paths that change shared state (slot lookups, CPU parts, predictions) run under this lock;
    // the large-batch copies (cached_expert, copy_experts) only read the slot maps of their own device
    std::mutex mtx_shared;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share,
            const std::string & profile, const llama_moe_cache_compute & compute) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
        const bool no_alloc = model.hparams.no_alloc || compute.measure;
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
                d.ib      = i;
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
        if (const char * env = getenv("LLAMA_MOE_RING")) {
            ring = std::max(0, atoi(env));
            if (ring > 0) {
                // a ring of one slot never runs ahead of the compute stream
                ring    = std::max(ring, 2);
                n_stage = ring;
                // the slots borrow the cache unless LLAMA_MOE_ASYNC_BORROW=0
                borrow  = getenv("LLAMA_MOE_ASYNC_BORROW") == nullptr || borrow;
            }
        }
        if (const char * env = getenv("LLAMA_MOE_RING_MERGE")) {
            ring_merge = std::max(0, atoi(env));
        }
        if (const char * env = getenv("LLAMA_MOE_RING_ZC")) {
            ring_zc = atoi(env);
        }
        if (const char * env = getenv("LLAMA_MOE_RING_BLOCKS")) {
            ring_blocks = std::max(1, atoi(env));
        }
        if (const char * env = getenv("LLAMA_MOE_RING_PRIO")) {
            ring_prio = atoi(env) != 0;
        }
        if (const char * env = getenv("LLAMA_MOE_RING_LOG")) {
            ring_log = atoi(env) != 0;
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
                ring    = 0;
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
        n_expert_model = (int32_t) model.hparams.n_expert;
        for (group & g : groups) {
            const device & d = devices[g.id];
            g.budget = (double) size*d.split/split_sum*g.host_bytes/d.host_bytes;
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_SPLIT")) {
            split_miss   = strcmp(env, "miss") == 0;
            split_stream = strcmp(env, "stream") == 0;
            if (!split_miss && !split_stream && *env != '\0' && strcmp(env, "free") != 0) {
                LLAMA_LOG_WARN("%s: unknown LLAMA_MOE_CACHE_SPLIT=%s (free, miss or stream), using free\n", __func__, env);
            }
        }
        if (split_miss && !compute.measure) {
            load_profile(profile);
            split_by_miss(size, compute, no_alloc);
        }
        if (split_stream && !compute.measure) {
            split_by_stream(size, compute, no_alloc);
        }
        apply_floors();
        // the plan without decode memory: the cache its large batches use is the least the plan with it may leave them
        size_slots();
        plan_memory();
        std::vector<double> budget_nodm;
        for (const group & g : groups) {
            budget_nodm.push_back(g.budget);
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            devices[id].pt_nodm = prompt_bytes_nodm(id, compute);
        }
        dm_setup(compute, no_alloc);
        if (dm_on) {
            apply_floors(); // over budget + lend
        }

        // size the groups, plan the lent tails and the staging slots; where the plan does not fit beside the compute buffers,
        // cut the device's budget by the excess and plan again
        for (int it = 0; ; ++it) {
            size_slots();
            plan_memory();
            if (no_alloc || it >= 8 || !fit(compute)) {
                break;
            }
        }
        dm_keep_prompt(budget_nodm);
        fit_report(compute);
        if (compute.measure) {
            // the small batch is measured on the cache path of every group (the floors), as the plan with decode memory has
            // them; a group below its floor would read whole expert tensors by op offload into the small compute buffer
            for (group & g : groups) {
                const int32_t n_expert = g.ref[0]->ne[2];
                const int32_t floor    = moe_cache_floor_slots(n_expert, n_used_batch(), (int32_t) g.layers.size()*n_expert);
                if (g.n_slots > 0 || floor >= n_expert_used) {
                    g.n_slots = std::max(g.n_slots, floor);
                }
            }
        }

        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const int32_t n_expert  = g.ref[0]->ne[2];
            if (g.n_slots == 0) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
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
        for (size_t id = 0; id < devices.size(); ++id) {
            ctx_views_dev.push_back(llama_moe_cache_init_ctx(2));
        }

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
            devices[g.id].buf_size += group_bytes(g, g.n_slots);
        }

        if (no_alloc) {
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
            for (size_t id = 0; id < devices.size(); ++id) {
                device & d = devices[id];
                if (!d.ctx) {
                    continue;
                }
                if (dm_on && dm_devs[id].lend > 0) {
                    dm_alloc(id);
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

        init_policy(profile, no_alloc);

        if (n_stage > 0) {
            init_stage(no_alloc);
        }
        if (!no_alloc && dm_on) {
            dm_alias_stage();
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
            if (d.host_bytes == 0 || compute.measure) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            if (d.slot_size > 0) {
                LLAMA_LOG_INFO("%s: %10s MoE staging   = %8.2f MiB, %d slots for %zu layers (LLAMA_MOE_ASYNC, all experts from %" PRId64 " ids per expert)\n", __func__,
                    ggml_backend_buft_name(d.buft), n_stage*d.slot_size/1024.0/1024.0, n_stage, d.layers.size(), full_ids);
            }
            if (d.mtx) {
                LLAMA_LOG_INFO("%s: %10s MoE ring: the %d staging slots stream every uncached expert of the device's layers ahead of "
                    "large batches (LLAMA_MOE_RING=%d, %s)\n", __func__, ggml_backend_buft_name(d.buft), n_stage, ring,
                    copy_blocks && ring_zc ? "one copy kernel per 128 experts, uncached ones read from mapped host memory" : "host copies per run");
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
                    if (g.lend_first < g.n_slots) {
                        LLAMA_LOG_INFO("%s:            %5d of them (%.1f%% of the layers' experts) lent to large batches\n", __func__,
                            g.n_slots - g.lend_first, 100.0*(g.n_slots - g.lend_first)/(g.layers.size()*g.ref[0]->ne[2]));
                    }
                }
            }
            if (id < dm_devs.size() && dm_devs[id].lend > 0) {
                const dm_device & dm = dm_devs[id];
                size_t rel = 0;
                for (const group & g : groups) {
                    if (g.id == (int32_t) id) {
                        for (const auto & r : g.lend_ranges) {
                            rel += r.second;
                        }
                    }
                }
                LLAMA_LOG_INFO("%s: %10s MoE decode memory: %.2f MiB of the cache back the compute buffer%s during large batches "
                    "(%.2f MiB measured, %.2f MiB kept for small ones, %.2f MiB staging; LLAMA_MOE_DECODE_MEM%s)\n", __func__,
                    ggml_backend_buft_name(d.buft), rel/1024.0/1024.0, dm.stage > 0 ? " and the staging slots" : "",
                    dm.large/1024.0/1024.0, dm.small/1024.0/1024.0, dm.stage/1024.0/1024.0,
                    dm_alias ? ", mapped there for good: a switch only evicts (LLAMA_MOE_DECODE_MEM_ALIAS)" : "");
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
        if (const char * env = getenv("LLAMA_MOE_CPU_LANE")) {
            lane_frac = std::clamp((float) atof(env), 0.0f, 1.0f);
        }
        if (const char * env = getenv("LLAMA_MOE_CPU_LANE_TIMEOUT_MS")) {
            lane_timeout_ms = std::max(1, atoi(env));
        }
        if (dh_mode > 0 && !no_alloc) {
            init_device_hits();
            if (lane_frac > 0.0f) {
                init_cpu_lane();
            }
        } else if (lane_frac > 0.0f && !no_alloc) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_CPU_LANE needs LLAMA_MOE_DEVICE_HITS, it is off\n", __func__);
            lane_frac = 0.0f;
        }
    }

    void init_policy(const std::string & profile_arg, bool no_alloc) {
        auto env_f = [](const char * name, double def) {
            const char * v = getenv(name);
            return v != nullptr && *v != '\0' ? atof(v) : def;
        };
        if (const char * env = getenv("LLAMA_MOE_CACHE_POLICY")) {
            ad_on = strcmp(env, "adaptive") == 0 || strcmp(env, "1") == 0;
            if (!ad_on && strcmp(env, "lru") != 0 && strcmp(env, "0") != 0 && *env != '\0') {
                LLAMA_LOG_WARN("%s: unknown LLAMA_MOE_CACHE_POLICY=%s (lru or adaptive), using lru\n", __func__, env);
            }
        }
        if (ad_on && cpu_share > 0.0f) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_POLICY=adaptive does not combine with --moe-cache-cpu-share, using lru\n", __func__);
            ad_on = false;
        }
        if (ad_on) {
            ad.init(layers.size()*(size_t) n_expert_model, env_f("LLAMA_MOE_CACHE_DECAY", 0.956), env_f("LLAMA_MOE_CACHE_MIN_GAIN", 1.5),
                env_f("LLAMA_MOE_CACHE_MIN_COUNT", 2.0));
            ad_fills_per_token = std::max(0, (int32_t) env_f("LLAMA_MOE_CACHE_FILLS_PER_TOKEN", 16));
            // enough for the misses of one decode token per plan (the slots are reused layer by layer); none with device hits,
            // whose misses are read from host memory
            const bool dh = getenv("LLAMA_MOE_DEVICE_HITS") != nullptr && atoi(getenv("LLAMA_MOE_DEVICE_HITS")) > 0;
            const int32_t want = std::max(0, (int32_t) env_f("LLAMA_MOE_CACHE_TRANSIENT", dh ? 0.0 : 2.0*n_expert_used));
            for (group & g : groups) {
                if (g.n_slots == 0) {
                    continue;
                }
                g.tr.n = std::min(want, g.n_slots/8);
                std::vector<int32_t> order;
                for (int32_t s = g.tr.n; s < g.n_slots; ++s) {
                    order.push_back(s);
                }
                moe_cache_slru_relink(g.slru, order);
            }
            ad_cands_g.resize(groups.size());
            LLAMA_LOG_INFO("%s: MoE cache policy: adaptive, decayed routing counts (x%.3f per graph), a miss is cached when its count "
                "is >= %.1f and beats the coldest cached expert's by %.2f, up to %d fills per token, %d transient slots per group\n",
                __func__, std::exp(-ad.k), ad.min_count, ad.gain, ad_fills_per_token, groups.empty() ? 0 : groups[0].tr.n);
        }

        load_profile(profile_arg);
        const bool exists = prof_exists;
        prof_keep     = std::clamp(env_f("LLAMA_MOE_CACHE_PROFILE_KEEP", 0.5), 0.0, 1.0);
        prof_save_min = std::max(0.0, env_f("LLAMA_MOE_CACHE_PROFILE_SAVE_MIN", 10.0));
        if (const char * env = getenv("LLAMA_MOE_CACHE_PROFILE_SAVE")) {
            prof_save = strcmp(env, "0") == 0 ? "" : env;
        } else if (!prof_path.empty()) {
            // Strata's file is never overwritten
            prof_save = prof.strata ? prof_path + ".lmcp" : prof_path;
        }
        if (!prof_save.empty() && !no_alloc) {
            prof_run.assign(layers.size()*(size_t) n_expert_model, 0.0);
            prof_saved_us = ggml_time_us();
        } else {
            prof_save.clear();
        }
        if (prof.n_layer > 0 && prof.n_layer != (int32_t) layers.size()) {
            LLAMA_LOG_WARN("%s: MoE cache profile %s has %d layers, the model %zu\n", __func__, prof_path.c_str(), prof.n_layer, layers.size());
        }
        if (prof.n_layer > 0 && !no_alloc) {
            place();
        } else if (!prof_path.empty() && !exists) {
            LLAMA_LOG_INFO("%s: MoE cache profile %s does not exist yet, the cache starts empty\n", __func__, prof_path.c_str());
        }
        if (!prof_save.empty()) {
            LLAMA_LOG_INFO("%s: MoE cache profile: the routings of this run are saved to %s (every %.0f min and at exit, %.0f%% of the loaded profile kept)\n",
                __func__, prof_save.c_str(), prof_save_min, 100.0*prof_keep);
        }
    }

    void load_profile(const std::string & profile_arg) {
        if (prof_loaded) {
            return;
        }
        prof_loaded = true;
        prof_path = profile_arg;
        if (prof_path.empty()) {
            if (const char * env = getenv("LLAMA_MOE_CACHE_PROFILE")) {
                prof_path = env;
            }
        }
        if (!prof_path.empty()) {
            FILE * f = fopen(prof_path.c_str(), "rb");
            prof_exists = f != nullptr;
            if (f) {
                fclose(f);
            }
            std::string err;
            if (prof_exists && !moe_cache_profile_read(prof_path, n_expert_model, prof, err)) {
                LLAMA_LOG_WARN("%s: MoE cache profile not used: %s\n", __func__, err.c_str());
                prof = moe_cache_profile();
            }
        }
    }

    // LLAMA_MOE_CACHE_SPLIT=miss: the budget goes to the experts with the highest profiled score first, whichever device holds
    // their layer (a miss costs its bytes, a slot costs the same bytes, so the gain per byte is the routing probability), as far
    // as each device has room beside its compute buffer; the rest (no profile, or experts without a score) by expert bytes
    void split_by_miss(size_t size, const llama_moe_cache_compute & compute, bool no_alloc) {
        const char * env_m = getenv("LLAMA_MOE_CACHE_SPLIT_MARGIN_MIB");
        const double margin = (env_m ? std::max(0.0, atof(env_m)) : 512.0)*1024.0*1024.0;
        std::vector<double> cap(devices.size(), 0.0);
        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            size_t free  = 0;
            size_t total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(d.backend), &free, &total);
            const double c = compute.large.size() > d.ib ? (double) compute.large[d.ib] : 0.0;
            cap[id] = no_alloc || free == 0 ? 1e30 : std::max(0.0, (double) free - c - margin);
        }
        if (compute.large.empty() && !no_alloc) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_SPLIT=miss without measured compute buffers: the devices' free memory caps the split\n", __func__);
        }

        struct item {
            float   score;
            int32_t ig;
            int32_t il;
            int32_t e;
        };
        std::vector<item> items;
        std::vector<double> bytes_g(groups.size(), 0.0);
        std::vector<double> mass_dev(devices.size(), 0.0);
        double mass_all = 0.0;
        const bool have = prof.n_layer == (int32_t) layers.size();
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            for (const ggml_tensor * t : g.ref) {
                bytes_g[ig] += (double) t->nb[2];
            }
            for (int32_t il : g.layers) {
                for (int32_t e = 0; e < (int32_t) g.ref[0]->ne[2] && have; ++e) {
                    const float sc = prof.get(il, e);
                    if (sc > 0.0f) {
                        items.push_back({ sc, (int32_t) ig, il, e });
                        mass_all += sc;
                    }
                }
            }
        }
        std::sort(items.begin(), items.end(), [](const item & a, const item & b) {
            return a.score != b.score ? a.score > b.score : (a.il != b.il ? a.il < b.il : a.e < b.e);
        });

        std::vector<double> got(groups.size(), 0.0);
        std::vector<double> used(devices.size(), 0.0);
        double left = (double) size;
        for (const item & it : items) {
            const double b  = bytes_g[it.ig];
            const int32_t id = groups[it.ig].id;
            if (b > left || used[id] + b > cap[id]) {
                continue;
            }
            got[it.ig] += b;
            used[id]   += b;
            left       -= b;
            mass_dev[id] += it.score;
        }
        // the rest by the bytes not cached yet, within the caps
        for (int round = 0; round < 8 && left >= 1.0; ++round) {
            double w_sum = 0.0;
            std::vector<double> w(groups.size(), 0.0);
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                const int32_t id = groups[ig].id;
                if (cap[id] - used[id] >= 1.0) {
                    w[ig]  = std::max(0.0, (double) groups[ig].host_bytes - got[ig]);
                    w_sum += w[ig];
                }
            }
            if (w_sum <= 0.0) {
                break;
            }
            double given = 0.0;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                const int32_t id = groups[ig].id;
                const double add = std::min(left*w[ig]/w_sum, cap[id] - used[id]);
                if (add > 0.0) {
                    got[ig]  += add;
                    used[id] += add;
                    given    += add;
                }
            }
            left -= given;
            if (given < 1.0) {
                break;
            }
        }
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            groups[ig].budget = got[ig];
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            if (devices[id].host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache split by misses: %8.2f MiB (cap %.2f MiB), %.1f%% of the profiled score%s\n", __func__,
                ggml_backend_buft_name(devices[id].buft), used[id]/1024.0/1024.0, cap[id] >= 1e29 ? 0.0 : cap[id]/1024.0/1024.0,
                mass_all > 0.0 ? 100.0*mass_dev[id]/mass_all : 0.0, have ? "" : " (no profile: same share of each device's experts)");
        }
        if (left >= 1024.0*1024.0) {
            LLAMA_LOG_WARN("%s: %.2f MiB of the MoE cache budget do not fit beside the compute buffers\n", __func__, left/1024.0/1024.0);
        }
    }

    // LLAMA_MOE_CACHE_SPLIT=stream: a piped prompt chunk streams every uncached expert of each stage, and the stage that streams
    // most sets the pace (srv-c3-allnodm: CUDA0 has 13 layers / 20.2 GB of host experts beside 11-12 / 17-18.3 GB, the same
    // cache, and ~2.3 GB of VRAM left unused). The budget goes where the most is streamed (moe_cache_split_stream), each device
    // capped by its room beside the measured compute buffer; --moe-cache-mib stays the total (a larger one is cut to the room).
    void split_by_stream(size_t size, const llama_moe_cache_compute & compute, bool no_alloc) {
        std::vector<double> need(devices.size(), 0.0);
        std::vector<double> cap(devices.size(), 0.0);
        std::vector<double> stage(devices.size(), 0.0);
        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            size_t slot_size = 0;
            const std::vector<size_t> sec = n_stage > 0 ? stage_sec(id, slot_size, true) : std::vector<size_t>();
            double own = 0.0;
            if (n_stage > 0 && borrow) {
                // the staging slots borrow the bank slots of the device's largest group during large batches
                int32_t ig_max = -1;
                for (size_t ig = 0; ig < groups.size(); ++ig) {
                    if (groups[ig].id == (int32_t) id && (ig_max < 0 || groups[ig].host_bytes > groups[ig_max].host_bytes)) {
                        ig_max = (int32_t) ig;
                    }
                }
                const std::vector<size_t> nb = bank_nb(groups[ig_max]);
                const int32_t k = moe_cache_stage_slots(nb, sec, n_stage);
                for (size_t b : nb) {
                    stage[id] += (double) k*b;
                }
            } else if (n_stage > 0) {
                own = (double) n_stage*slot_size;
            }
            need[id] = (double) d.host_bytes + stage[id];
            size_t free  = 0;
            size_t total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(d.backend), &free, &total);
            const double large   = compute.large.size()   > d.ib ? (double) compute.large[d.ib]   : 0.0;
            const double reserve = compute.reserve.size() > d.ib ? (double) compute.reserve[d.ib] : 0.0;
            cap[id] = no_alloc || free == 0 ? 1e30 : std::max(0.0, (double) free - fit_margin(id) - large - reserve - own);
        }
        if (compute.large.empty() && !no_alloc) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_SPLIT=stream without measured compute buffers: the devices' free memory caps the split\n", __func__);
        }
        std::vector<double> b;
        moe_cache_split_stream(need, cap, (double) size, b);
        double given = 0.0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            g.budget = d.host_bytes > 0 ? b[g.id]*g.host_bytes/d.host_bytes : 0.0;
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            if (devices[id].host_bytes == 0) {
                continue;
            }
            given += b[id];
            LLAMA_LOG_INFO("%s: %10s MoE cache split by streamed bytes: %8.2f MiB (cap %.2f MiB); a prompt chunk streams %.2f MiB "
                "(%.2f MiB host experts, %.2f MiB staging borrowed)\n", __func__, ggml_backend_buft_name(devices[id].buft),
                b[id]/1024.0/1024.0, cap[id] >= 1e29 ? 0.0 : cap[id]/1024.0/1024.0, (need[id] - b[id])/1024.0/1024.0,
                devices[id].host_bytes/1024.0/1024.0, stage[id]/1024.0/1024.0);
        }
        if (given + 1024.0*1024.0 < (double) size) {
            LLAMA_LOG_WARN("%s: %.2f MiB of the MoE cache budget do not fit beside the compute buffers (or are not needed)\n", __func__,
                ((double) size - given)/1024.0/1024.0);
        }
    }

    // LLAMA_MOE_DECODE_MEM: which devices lend, and how much (the part of the compute buffer that small batches do not use)
    void dm_setup(const llama_moe_cache_compute & compute, bool no_alloc) {
        dm_devs.assign(devices.size(), dm_device());
        const char * env = getenv("LLAMA_MOE_DECODE_MEM");
        if (env == nullptr || atoi(env) == 0 || no_alloc) {
            return;
        }
        if (borrow && ring == 0) {
            // the ring's borrowed slots sit below the lent tail (plan_borrow); the per-layer staging's have no such placement
            LLAMA_LOG_WARN("%s: LLAMA_MOE_DECODE_MEM does not combine with LLAMA_MOE_ASYNC_BORROW without LLAMA_MOE_RING, it is off\n", __func__);
            return;
        }
        if (compute.large.empty()) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_DECODE_MEM needs the measured compute buffers, it is off\n", __func__);
            return;
        }
        auto env_i = [](const char * name, int def) {
            const char * v = getenv(name);
            return v != nullptr && *v != '\0' ? atoi(v) : def;
        };
        dm_return_after = std::max(1, env_i("LLAMA_MOE_DECODE_MEM_RETURN", 1));
        dm_refill_cap   = std::max(0, env_i("LLAMA_MOE_DECODE_MEM_REFILL", 256));
        const size_t spare = (size_t) std::max(0, env_i("LLAMA_MOE_DECODE_MEM_SPARE_MIB", 64))*1024*1024;
        if (const char * v = getenv("LLAMA_MOE_DECODE_MEM_COMPACT")) {
            dm_compact = std::max(0.0f, (float) atof(v));
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            dm_device & dm = dm_devs[id];
            if (d.host_bytes == 0 || compute.large.size() <= d.ib || compute.small.size() <= d.ib) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            auto alloc   = (vmm_alloc_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_alloc");
            auto commit  = (vmm_commit_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_commit");
            auto release = (vmm_release_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_release");
            auto gran    = (vmm_gran_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_get_granularity");
            auto cbuft   = (vmm_cbuft_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_vmm_compute_buffer_type");
            ggml_backend_buffer_type_t cb = cbuft ? cbuft(dev) : nullptr;
            const size_t g = gran ? gran(d.buft) : 0;
            if (!alloc || !commit || !release || g == 0 || cb == nullptr) {
                LLAMA_LOG_WARN("%s: %s has no virtual memory buffers, LLAMA_MOE_DECODE_MEM is off there\n", __func__, ggml_backend_name(d.backend));
                continue;
            }
            vmm_alloc   = alloc;
            vmm_commit  = commit;
            vmm_release = release;
            vmm_release_spare = (vmm_release_spare_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_release_spare");
            vmm_commit_spare  = (vmm_commit_spare_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_commit_spare");
            vmm_trim          = (vmm_trim_t)          ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_spare_trim");
            if (!vmm_release_spare || !vmm_commit_spare || !vmm_trim) {
                vmm_release_spare = nullptr;
                vmm_commit_spare  = nullptr;
                vmm_trim          = nullptr;
            }
            vmm_chunk    = (vmm_commit_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_commit_chunk");
            vmm_alias_fn = (vmm_alias_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_alias");
            dm_alias     = vmm_chunk && vmm_alias_fn && env_i("LLAMA_MOE_DECODE_MEM_ALIAS", 1) != 0;
            dm.gran  = g;
            dm.cbuft = cb;
            dm.large = compute.large[d.ib];
            dm.small = GGML_PAD(compute.small[d.ib] + spare, g);
            if (dm.large > dm.small + 16*g) {
                dm.lend = GGML_PAD(dm.large - dm.small, g);
            }
            if (ring > 0 && dm.lend > 0) {
                // the ring's staging slots are used by large batches only: they live in the memory the cache lends to them
                size_t slot_size = 0;
                stage_sec(id, slot_size, true);
                dm.stage = GGML_PAD(n_stage*slot_size, g);
                dm.lend += dm.stage;
            }
            if (dm.lend == 0) {
                continue;
            }
            dm_on = true;
            // each group lends in proportion to its budget
            double sum = 0.0;
            for (const group & gr : groups) {
                sum += gr.id == (int32_t) id ? gr.budget : 0.0;
            }
            for (group & gr : groups) {
                if (gr.id == (int32_t) id) {
                    gr.lend = sum > 0.0 ? (double) dm.lend*gr.budget/sum : (double) dm.lend*gr.host_bytes/d.host_bytes;
                }
            }
        }
    }

    // bytes of a group of n slots as the budget counts them (the n_zero slots and one extra after them included)
    size_t group_bytes(const group & g, int32_t n) const {
        const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
        size_t res = 0;
        for (const ggml_tensor * t : g.ref) {
            res += GGML_PAD(t->nb[2]*(n + n_zero + 1), alignment);
        }
        return res;
    }

    // allocation size of a tensor of n experts like t on the device (the backend may pad it)
    size_t expert_alloc_size(const device & d, const ggml_tensor * t, int64_t n) const {
        ggml_context_ptr ctx = llama_moe_cache_init_ctx(1);
        ggml_tensor * x = ggml_new_tensor_3d(ctx.get(), t->type, t->ne[0], t->ne[1], n);
        return ggml_backend_buft_get_alloc_size(d.buft, x);
    }

    int64_t n_used_batch() const {
        return max_batch*n_expert_used;
    }

    // the transient slots init_policy will give a group of n slots (LLAMA_MOE_CACHE_POLICY=adaptive without device hits)
    int32_t transient_slots(int32_t n) const {
        const char * pol = getenv("LLAMA_MOE_CACHE_POLICY");
        const bool ad = pol != nullptr && (strcmp(pol, "adaptive") == 0 || strcmp(pol, "1") == 0) && cpu_share <= 0.0f;
        if (!ad) {
            return 0;
        }
        const char * dh_env = getenv("LLAMA_MOE_DEVICE_HITS");
        const bool dh = dh_env != nullptr && atoi(dh_env) > 0;
        const char * tr_env = getenv("LLAMA_MOE_CACHE_TRANSIENT");
        const int32_t want = std::max(0, tr_env != nullptr && *tr_env != '\0' ? atoi(tr_env) : (dh ? 0 : 2*n_expert_used));
        return std::min(want, n/8);
    }

    // every group gets the slots for the cache path of max_batch tokens, from the larger groups of its device: a group of one
    // layer of another quantization gets a small share by bytes, and below that floor graphs of up to max_batch tokens read its
    // layers' host experts by op offload (whole expert tensors copied into the compute buffer of the small batches)
    void apply_floors() {
        for (size_t id = 0; id < devices.size(); ++id) {
            std::vector<size_t> igs;
            std::vector<double> budget;
            std::vector<double> lend;
            std::vector<double> floor;
            std::vector<double> cap;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                const group & g = groups[ig];
                if (g.id != (int32_t) id) {
                    continue;
                }
                const int32_t n_expert = g.ref[0]->ne[2];
                igs.push_back(ig);
                budget.push_back(g.budget);
                lend.push_back(g.lend);
                floor.push_back((double) group_bytes(g, moe_cache_floor_slots(n_expert, n_used_batch(), (int32_t) g.layers.size()*n_expert)));
                cap.push_back((double) group_bytes(g, (int32_t) g.layers.size()*n_expert));
            }
            const bool lends = dm_on && id < dm_devs.size() && dm_devs[id].lend > 0;
            if (lends ? !igs.empty() && moe_cache_plan_floors_dm(budget, lend, floor, cap) : igs.size() > 1 && moe_cache_plan_floors(budget, floor)) {
                for (size_t i = 0; i < igs.size(); ++i) {
                    groups[igs[i]].budget = budget[i];
                    groups[igs[i]].lend   = lend[i];
                }
            }
        }
    }

    void size_slots() {
        for (group & g : groups) {
            const int32_t n_expert  = g.ref[0]->ne[2];
            const double  budget    = std::max(0.0, g.budget + g.lend);
            const int32_t max_slots = g.layers.size()*n_expert;
            // binary search: the largest n with group_bytes(n) <= budget
            int32_t lo = 0;
            int32_t hi = max_slots;
            while (lo < hi) {
                const int32_t mid = lo + (hi - lo + 1)/2;
                if ((double) group_bytes(g, mid) <= budget) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            g.n_slots    = lo >= n_expert_used ? lo : 0;
            g.lend_first = g.n_slots;
            g.lend_ranges.clear();
            g.bank_offs.clear();
        }
    }

    // [expert tensor] the largest staging tensor of the device's groups (the slot spacing in a borrowed bank tail), `slot_size`
    // the largest staging slot of one group (a slot of their own); as init_stage lays them out
    std::vector<size_t> stage_sec(size_t id, size_t & slot_size, bool all = false) const {
        const device & d = devices[id];
        const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
        std::vector<size_t> sec;
        slot_size = 0;
        for (const group & g : groups) {
            if (g.id != (int32_t) id || (g.n_slots == 0 && !all)) {
                continue;
            }
            size_t size = 0;
            for (size_t ip = 0; ip < g.ref.size(); ++ip) {
                const size_t st = GGML_PAD(expert_alloc_size(d, g.ref[ip], g.ref[ip]->ne[2] + 1), alignment);
                size += st;
                sec.resize(std::max(sec.size(), ip + 1), 0);
                sec[ip] = std::max(sec[ip], st);
            }
            slot_size = std::max(slot_size, size);
        }
        return sec;
    }

    std::vector<size_t> bank_nb(const group & g) const {
        std::vector<size_t> nb;
        for (const ggml_tensor * t : g.ref) {
            nb.push_back(t->nb[2]);
        }
        return nb;
    }

    // LLAMA_MOE_CACHE_MARGIN_MIB, 512: one value for every device, or a comma list by device (e.g. room for an MTP draft
    // context loaded after this one)
    double fit_margin(size_t id) const {
        const char * env_m = getenv("LLAMA_MOE_CACHE_MARGIN_MIB");
        if (env_m == nullptr) {
            env_m = getenv("LLAMA_MOE_CACHE_SPLIT_MARGIN_MIB");
        }
        std::vector<double> margins;
        for (const char * p = env_m; p != nullptr && *p != '\0'; ) {
            margins.push_back(std::max(0.0, atof(p))*1024.0*1024.0);
            p = strchr(p, ',');
            p = p ? p + 1 : nullptr;
        }
        return margins.empty() ? 512.0*1024.0*1024.0 : margins[std::min(id, margins.size() - 1)];
    }

    // the cache large batches use without decode memory (plan_memory without lends): at most what fits beside the compute
    // buffer of -ub and the staging slots of their own (that plan does not fit itself to the device)
    size_t prompt_bytes_nodm(size_t id, const llama_moe_cache_compute & compute) const {
        const device & d = devices[id];
        size_t pt = prompt_bytes(id);
        if (compute.large.size() <= d.ib) {
            return pt;
        }
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(d.backend), &free, &total);
        if (free == 0) {
            return pt;
        }
        size_t slots = 0; // the whole cache: the staging slots it lends are out of pt
        for (const group & g : groups) {
            if (g.id == (int32_t) id && g.n_slots > 0) {
                for (size_t nb : bank_nb(g)) {
                    slots += nb*(size_t) g.n_slots;
                }
            }
        }
        const double avail = (double) free - fit_margin(id) - (double) (compute.reserve.size() > d.ib ? compute.reserve[d.ib] : 0) -
            (double) compute.large[d.ib] - (double) d.plan_stage_own - (double) (slots - std::min(slots, pt));
        return std::min(pt, (size_t) std::max(0.0, avail));
    }

    // the device's cache that large batches use as planned (plan_memory), see moe_cache_plan_prompt_bytes
    size_t prompt_bytes(size_t id) const {
        const device & d = devices[id];
        std::vector<size_t>  per_slot;
        std::vector<int32_t> first;
        int32_t lender = -1;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            if (g.id != (int32_t) id || g.n_slots == 0) {
                continue;
            }
            size_t ps = 0;
            for (size_t nb : bank_nb(g)) {
                ps += nb;
            }
            if ((int32_t) ig == d.plan_lender) {
                lender = (int32_t) per_slot.size();
            }
            per_slot.push_back(ps);
            first.push_back(std::min(g.lend_first, g.n_slots));
        }
        return moe_cache_plan_prompt_bytes(per_slot, first, lender, d.plan_stage_k);
    }

    // decode memory only adds to the cache of small batches: a device whose plan with it leaves large batches less cache than
    // the plan without it (the slots kept below the tails, a budget cut to fit the compute buffer of -ub beside it) plans
    // without it
    void dm_keep_prompt(const std::vector<double> & budget_nodm) {
        bool any = false;
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (!dm_on || id >= dm_devs.size() || dm_devs[id].lend == 0 || d.host_bytes == 0) {
                continue;
            }
            size_t tol = 32*1024*1024; // whole slots and granules, fit()'s extra 16 MiB per cut
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    for (size_t nb : bank_nb(g)) {
                        tol += nb + dm_devs[id].gran;
                    }
                }
            }
            const size_t pt = prompt_bytes(id);
            if (pt + tol >= d.pt_nodm) {
                continue;
            }
            LLAMA_LOG_WARN("%s: %s: with LLAMA_MOE_DECODE_MEM large batches would use %.2f MiB of the MoE cache, less than the %.2f MiB "
                "without it: decode memory is off on this device\n", __func__, ggml_backend_name(d.backend), pt/1024.0/1024.0,
                d.pt_nodm/1024.0/1024.0);
            dm_devs[id].lend  = 0;
            dm_devs[id].stage = 0;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                if (groups[ig].id == (int32_t) id) {
                    groups[ig].budget = budget_nodm[ig];
                    groups[ig].lend   = 0.0;
                }
            }
            d.fit_cut   = 0.0;
            d.fit_stuck = false;
            any = true;
        }
        if (!any) {
            return;
        }
        dm_on = false;
        for (const dm_device & dm : dm_devs) {
            dm_on = dm_on || dm.lend > 0;
        }
        size_slots();
        plan_memory();
    }

    // per device: the group that holds the staging slots of LLAMA_MOE_RING (largest first, one with room for them above the slots
    // it keeps), the lent tails of LLAMA_MOE_DECODE_MEM below which that group keeps the staging slots, the buffer sizes
    void plan_memory() {
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            d.plan_lender    = -1;
            d.plan_stage_k   = 0;
            d.plan_total     = 0;
            d.plan_lent      = 0;
            d.plan_stage_own = 0;
            if (d.host_bytes == 0) {
                continue;
            }
            const bool lends = dm_on && id < dm_devs.size() && dm_devs[id].lend > 0;

            if (n_stage > 0 && !(lends && dm_devs[id].stage > 0)) {
                size_t slot_size = 0;
                const std::vector<size_t> sec = stage_sec(id, slot_size);
                std::vector<int32_t> cand;
                bool same_banks = true;
                for (size_t ig = 0; ig < groups.size(); ++ig) {
                    if (groups[ig].id == (int32_t) id && groups[ig].n_slots > 0) {
                        cand.push_back((int32_t) ig);
                        same_banks = same_banks && groups[ig].ref.size() == sec.size();
                    }
                }
                std::stable_sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return groups[a].n_slots > groups[b].n_slots; });
                for (int32_t ig : cand) {
                    if (!borrow || !same_banks) {
                        break;
                    }
                    const group & g = groups[ig];
                    const int32_t k = moe_cache_stage_slots(bank_nb(g), sec, n_stage);
                    // lending: room for the slots kept, the staging slots and one for the alignment of their start below the tail;
                    // otherwise the staging slots end at the last slot (plan_borrow)
                    const int32_t need = lends ? moe_cache_keep_slots(g.ref[0]->ne[2], n_used_batch(), transient_slots(g.n_slots)) + k + 1 : k + 1;
                    if (g.n_slots >= need) {
                        d.plan_lender  = ig;
                        d.plan_stage_k = k;
                        break;
                    }
                }
                if (d.plan_lender < 0 && !cand.empty()) {
                    d.plan_stage_own = n_stage*slot_size;
                }
            }

            if (!lends) {
                for (const group & g : groups) {
                    if (g.id == (int32_t) id && g.n_slots > 0) {
                        d.plan_total += group_bytes(g, g.n_slots);
                    }
                }
                continue;
            }

            std::vector<size_t> igs;
            std::vector<moe_cache_plan_group> pg;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                const group & g = groups[ig];
                if (g.id != (int32_t) id) {
                    continue;
                }
                moe_cache_plan_group p;
                if (g.n_slots > 0) {
                    p.nb = bank_nb(g);
                    for (const ggml_tensor * t : g.ref) {
                        p.size.push_back(expert_alloc_size(d, t, g.n_slots + n_zero + 1));
                    }
                }
                p.n_slots   = g.n_slots;
                p.min_first = moe_cache_keep_slots(g.ref[0]->ne[2], n_used_batch(), transient_slots(g.n_slots)) +
                    ((int32_t) ig == d.plan_lender ? d.plan_stage_k + 1 : 0);
                p.lend      = g.lend;
                igs.push_back(ig);
                pg.push_back(std::move(p));
            }
            d.plan_total = moe_cache_plan_lend(pg, dm_devs[id].gran, d.plan_lent);
            for (size_t i = 0; i < igs.size(); ++i) {
                group & g = groups[igs[i]];
                g.bank_offs   = pg[i].off;
                g.lend_first  = pg[i].first;
                g.lend_ranges = pg[i].ranges;
            }
        }
    }

    // the planned peak of each device against its free memory (with a margin for the CUDA pools and contexts loaded later,
    // LLAMA_MOE_CACHE_MARGIN_MIB, 512, or a comma list by device);
    // a device over it has its budget cut by the excess, true if any was cut (plan again). A cut that did not lower the peak
    // (the slots kept under the lent tails and the staging slots, not the budget, set it) is undone and the device left as it is.
    bool fit(const llama_moe_cache_compute & compute) {
        if (compute.large.empty()) {
            return false;
        }
        auto margin_of = [this](size_t id) { return fit_margin(id); };
        bool again = false;
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (d.host_bytes == 0 || d.fit_stuck || compute.large.size() <= d.ib || compute.small.size() <= d.ib) {
                continue;
            }
            size_t free  = 0;
            size_t total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(d.backend), &free, &total);
            if (free == 0) {
                continue;
            }
            const bool lends = dm_on && id < dm_devs.size() && dm_devs[id].lend > 0;
            const size_t small = lends ? dm_devs[id].small : compute.small[d.ib];
            const size_t stage_lent = lends ? dm_devs[id].stage : 0;
            const double peak = (double) moe_cache_plan_peak(d.plan_total, d.plan_lent, d.plan_stage_own, stage_lent, compute.large[d.ib], small, lends) + margin_of(id) +
                (double) (compute.reserve.size() > d.ib ? compute.reserve[d.ib] : 0);
            const double excess = peak - (double) free;
            if (excess <= 0.0) {
                continue;
            }
            // a budget beyond all of a group's experts buys nothing: cut from what is used
            std::vector<size_t> igs;
            double sum = 0.0;
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                group & g = groups[ig];
                if (g.id == (int32_t) id) {
                    const int32_t n_expert  = g.ref[0]->ne[2];
                    const int32_t max_slots = (int32_t) g.layers.size()*n_expert;
                    const double  max_bytes = (double) group_bytes(g, max_slots);
                    const double  floor     = (double) group_bytes(g, moe_cache_floor_slots(n_expert, n_used_batch(), max_slots));
                    g.budget = std::min(g.budget, std::max(floor, max_bytes - g.lend));
                    igs.push_back(ig);
                    sum += g.budget;
                }
            }
            const bool no_help = d.fit_excess > 0.0 && d.fit_excess - excess < std::min(64.0*1024.0*1024.0, d.fit_last_cut/4);
            if (no_help || sum <= 0.0) {
                // the last cut did not help: undo it; or nothing is left to cut; the device stays over
                if (!no_help) {
                    d.fit_saved.clear();
                }
                for (const auto & sv : d.fit_saved) {
                    groups[sv.first].budget = sv.second;
                }
                LLAMA_LOG_WARN("%s: %s: the MoE cache plan needs %.2f MiB more than the device has free, and a smaller cache does not "
                    "lower that: the compute buffer of -ub, the staging slots, the slots kept below the lent tails and the memory left "
                    "for a draft / MTP context (--cpu-moe-draft keeps its experts in host memory) set it\n",
                    __func__, ggml_backend_name(d.backend), excess/1024.0/1024.0);
                d.fit_cut  -= d.fit_saved.empty() ? 0.0 : d.fit_last_cut;
                d.fit_stuck = true;
                again = !d.fit_saved.empty();
                continue;
            }
            d.fit_saved.clear();
            for (size_t ig : igs) {
                d.fit_saved.push_back({ ig, groups[ig].budget });
            }
            const double cut = std::min(excess + 16.0*1024.0*1024.0, sum); // a little more: whole slots and granules
            for (size_t ig : igs) {
                groups[ig].budget *= (sum - cut)/sum;
            }
            d.fit_cut     += cut;
            d.fit_last_cut = cut;
            d.fit_excess   = excess;
            again = true;
        }
        if (again) {
            apply_floors();
        }
        return again;
    }

    void fit_report(const llama_moe_cache_compute & compute) const {
        for (const device & d : devices) {
            if (compute.reserve.size() > d.ib && compute.reserve[d.ib] > 0 && d.host_bytes > 0) {
                LLAMA_LOG_INFO("%s: %s: the MoE cache plan leaves %.2f MiB free for a context created later (draft / MTP)\n", __func__,
                    ggml_backend_name(d.backend), compute.reserve[d.ib]/1024.0/1024.0);
            }
            if (d.fit_cut > 0.0) {
                LLAMA_LOG_WARN("%s: %s: the MoE cache is %.2f MiB smaller than its share of --moe-cache-mib, so that the cache, its lent "
                    "tail, the staging slots and the compute buffers fit the device's free memory\n", __func__,
                    ggml_backend_name(d.backend), d.fit_cut/1024.0/1024.0);
            }
        }
    }

    // the banks of a lending device in a VMM buffer as planned (plan_memory): each bank on a granule, the tails lent in whole granules
    void dm_alloc(size_t id) {
        device & d = devices[id];
        dm_device & dm = dm_devs[id];
        if (d.plan_lent < dm.lend) {
            LLAMA_LOG_WARN("%s: %s lends %.2f MiB of the %.2f MiB the large batches need beside the small ones, the rest is new memory during large batches\n",
                __func__, ggml_backend_name(d.backend), d.plan_lent/1024.0/1024.0, dm.lend/1024.0/1024.0);
        }

        const size_t total = d.plan_total;
        d.buf.reset(vmm_alloc(d.buft, total));
        if (!d.buf) {
            throw std::runtime_error("failed to reserve the address space of the MoE cache");
        }
        dm.tails.clear();
        for (const group & g : groups) {
            if (g.id == (int32_t) id) {
                for (const auto & r : g.lend_ranges) {
                    // aliased: one allocation per lent range, so that each maps with one call
                    if (dm_alias && !vmm_chunk(d.buf.get(), r.first, r.second)) {
                        throw std::runtime_error("failed to allocate the MoE cache buffers");
                    }
                    dm.tails.push_back(r);
                }
            }
        }
        if (!vmm_commit(d.buf.get(), 0, total)) {
            throw std::runtime_error("failed to allocate the MoE cache buffers");
        }
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(d.buf.get());
        for (group & g : groups) {
            if (g.id != (int32_t) id || g.n_slots == 0) {
                continue;
            }
            GGML_ASSERT(g.bank_offs.size() == g.banks.size());
            for (size_t ip = 0; ip < g.banks.size(); ++ip) {
                GGML_ASSERT(g.bank_offs[ip] + ggml_backend_buft_get_alloc_size(d.buft, g.banks[ip]) <= total);
                if (ggml_backend_tensor_alloc(d.buf.get(), g.banks[ip], base + g.bank_offs[ip]) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("failed to place the MoE cache banks");
                }
            }
        }
        for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
            if (t->view_src != nullptr && t->buffer == nullptr) {
                ggml_backend_view_init(t);
            }
        }
        d.buf_size = total;
    }

    // LLAMA_MOE_DECODE_MEM_ALIAS: map [t_off, t_off + size) of the device's tail space (its lent ranges in a row) into dst at dst_off
    bool dm_alias_map(size_t id, ggml_backend_buffer_t dst, size_t dst_off, size_t t_off, size_t size) {
        const dm_device & dm = dm_devs[id];
        size_t pos = 0;
        for (const auto & r : dm.tails) {
            if (size == 0) {
                break;
            }
            if (t_off >= pos + r.second) {
                pos += r.second;
                continue;
            }
            const size_t skip = t_off - pos;
            const size_t n    = std::min(size, r.second - skip);
            if (!vmm_alias_fn(dst, dst_off, devices[id].buf.get(), r.first + skip, n)) {
                return false;
            }
            dst_off += n;
            t_off   += n;
            size    -= n;
            pos     += r.second;
        }
        return size == 0;
    }

    size_t dm_tail_bytes(size_t id) const {
        size_t res = 0;
        for (const auto & r : dm_devs[id].tails) {
            res += r.second;
        }
        return res;
    }

    // the staging slots on the first part of the tail space, once (after init_stage cleared their buffer)
    void dm_alias_stage() {
        for (size_t id = 0; id < devices.size() && dm_alias; ++id) {
            device & d = devices[id];
            dm_device & dm = dm_devs[id];
            if (dm.lend == 0 || dm.stage == 0 || !d.buf_stage) {
                continue;
            }
            const size_t n = std::min(dm.stage, dm_tail_bytes(id))/dm.gran*dm.gran;
            if (!dm_alias_map(id, d.buf_stage.get(), 0, 0, n)) {
                throw std::runtime_error("failed to map the MoE staging slots onto the lent cache tails");
            }
            dm.alias_stage = n;
        }
    }

    // a new compute buffer: the small part of its own, the rest of the tail space after it
    void dm_alias_compute(size_t id) {
        dm_device & dm = dm_devs[id];
        const size_t bsize = ggml_backend_buffer_get_size(dm.cbuf);
        dm.alias_comp = 0;
        if (bsize <= dm.small) {
            return;
        }
        dm_commit_compute(id, dm.small);
        const size_t left = dm_tail_bytes(id) - std::min(dm_tail_bytes(id), dm.alias_stage);
        const size_t n    = std::min(left, bsize - dm.small)/dm.gran*dm.gran;
        if (n > 0 && !dm_alias_map(id, dm.cbuf, dm.small, dm.alias_stage, n)) {
            GGML_ABORT("%s: %s: cannot map the compute buffer onto the lent cache tails (LLAMA_MOE_DECODE_MEM_ALIAS=0 moves them instead)",
                __func__, ggml_backend_name(devices[id].backend));
        }
        dm.alias_comp = n;
        dm.committed  = dm.small + n;
    }

    // fill each group's slots with the hottest experts of its layers by the profile, before the first graph
    void place() {
        std::vector<bool> used_dev(devices.size(), false);
        std::vector<size_t> placed(devices.size(), 0);
        std::vector<size_t> slots_dev(devices.size(), 0);
        std::vector<double> mass(devices.size(), 0.0);
        std::vector<double> mass_all(devices.size(), 0.0);
        std::vector<size_t> bytes(devices.size(), 0);
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            moe_cache_slru & c = g.slru;
            const int32_t first = g.tr.n;
            const auto picks = moe_cache_profile_pick(prof, g.layers, c.n_expert, (size_t) (g.n_slots - first));
            std::vector<std::pair<float, int32_t>> order; // (score, slot) of the placed experts
            for (size_t i = 0; i < picks.size(); ++i) {
                const int32_t s  = first + (int32_t) i;
                const int32_t il = picks[i].first;
                const int32_t e  = picks[i].second;
                c.key_of[s] = il*c.n_expert + e;
                c.slot_map[il][e] = s;
                c.n_filled++;
                order.push_back({ prof.get(il, e), s });
                mass[g.id] += prof.get(il, e);
            }
            for (int32_t il : g.layers) {
                for (int32_t e = 0; e < c.n_expert; ++e) {
                    mass_all[g.id] += prof.get(il, e);
                }
            }
            // the free slots are the next victims, then the placed experts coldest first
            std::sort(order.begin(), order.end());
            std::vector<int32_t> lst;
            for (int32_t s = first + (int32_t) picks.size(); s < g.n_slots; ++s) {
                lst.push_back(s);
            }
            for (const auto & o : order) {
                lst.push_back(o.second);
            }
            moe_cache_slru_relink(c, lst);

            // consecutive experts of a layer are in consecutive slots: one upload per run
            ggml_backend_t backend = devices[g.id].backend;
            for (size_t i = 0; i < picks.size(); ) {
                size_t n = 1;
                while (i + n < picks.size() && picks[i + n].first == picks[i].first && picks[i + n].second == picks[i].second + (int32_t) n) {
                    n++;
                }
                const layer & l = layers[picks[i].first];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t es = w->nb[2];
                    ggml_backend_tensor_set_async(backend, g.banks[ip], (const uint8_t *) w->data + picks[i].second*es, (first + i)*es, n*es);
                    bytes[g.id] += n*es;
                }
                i += n;
            }
            used_dev[g.id] = true;
            placed[g.id]    += picks.size();
            slots_dev[g.id] += g.n_slots - first;
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            if (!used_dev[id]) {
                continue;
            }
            ggml_backend_synchronize(devices[id].backend);
            LLAMA_LOG_INFO("%s: %10s MoE cache profile: %zu of %zu slots filled with the hottest experts of its layers (%.2f MiB), "
                "%.1f%% of their profiled %s\n", __func__, ggml_backend_name(devices[id].backend), placed[id], slots_dev[id],
                bytes[id]/1024.0/1024.0, mass_all[id] > 0.0 ? 100.0*mass[id]/mass_all[id] : 0.0,
                prof.strata ? "rank mass (Strata profile)" : "routings");
        }
    }

    // count the routings of a small batch: the profile and the adaptive policy
    void observe(int32_t il, const int32_t * ids, size_t n) {
        const size_t base = (size_t) il*n_expert_model;
        for (size_t i = 0; i < n; ++i) {
            if (ids[i] < 0 || ids[i] >= n_expert_model) {
                continue;
            }
            if (!prof_run.empty()) {
                prof_run[base + ids[i]] += 1.0;
            }
            if (ad_on) {
                ad.add((int32_t) (base + ids[i]), 1.0);
            }
        }
        if (!prof_run.empty()) {
            prof_run_sum += (double) n;
        }
    }

    void profile_save(bool at_exit) {
        if (prof_save.empty() || prof_run_sum <= prof_saved_sum) {
            return;
        }
        const size_t n = prof_run.size();
        double sum_prior = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum_prior += prof.n_layer > 0 ? prof.get((int32_t) (i / n_expert_model), (int32_t) (i % n_expert_model)) : 0.0f;
        }
        const double keep = sum_prior > 0.0 ? prof_keep : 0.0;
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) {
            const double prior = sum_prior > 0.0 ? prof.get((int32_t) (i / n_expert_model), (int32_t) (i % n_expert_model))/sum_prior : 0.0;
            out[i] = (float) (1e6*(keep*prior + (1.0 - keep)*prof_run[i]/prof_run_sum));
        }
        std::string err;
        if (moe_cache_profile_write(prof_save, (int32_t) layers.size(), n_expert_model, out, err)) {
            prof_saved_sum = prof_run_sum;
            LLAMA_LOG_INFO("llama_moe_cache: profile saved to %s (%.0f routings this run%s)\n", prof_save.c_str(), prof_run_sum, at_exit ? ", at exit" : "");
        } else {
            LLAMA_LOG_WARN("llama_moe_cache: profile not saved: %s\n", err.c_str());
        }
        prof_saved_us = ggml_time_us();
    }

    void clear_transient(group & g) {
        moe_cache_transient_clear(g.slru, g.tr);
    }

    bool plan_adaptive(group & g, int32_t il, const int32_t * ids_p, size_t n_ids, int32_t n_used, size_t & n_hit) {
        if (ad_budget_tick != ad.tick) {
            ad_budget_tick = ad.tick;
            ad_budget = (int64_t) ad_fills_per_token*std::max<int64_t>(1, (int64_t) n_ids/std::max(1, n_used));
        }
        cpu.clear();
        return moe_cache_plan_adaptive(g.slru, g.tr, ad, ad_budget, il, ids_p, n_ids, fills, n_hit, stats_ad, ad_scratch);
    }

    // one channel per device-path layer, in a pinned host buffer per device mapped to the device
    void init_cpu_lane() {
        int64_t n_embd = 0; // an upper bound of the rows of the experts' input and output
        for (const layer & l : layers) {
            for (const ggml_tensor * w : l.experts) {
                n_embd = std::max({ n_embd, w->ne[0], w->ne[1] });
            }
        }
        const size_t chan_size = ggml_moe_lane_layout(n_embd, n_expert_used, dh_max, nullptr, nullptr, nullptr);

        size_t n_chan = 0;
        for (size_t id = 0; id < dh_devs.size(); ++id) {
            const dh_device & dd = dh_devs[id];
            if (dd.rows.empty()) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(devices[id].backend);
            {
                ggml_context_ptr sctx = llama_moe_cache_init_ctx(8);
                ggml_tensor * ids = ggml_new_tensor_2d(sctx.get(), GGML_TYPE_I32, n_expert_used, dh_max);
                ggml_tensor * x   = ggml_new_tensor_3d(sctx.get(), GGML_TYPE_F32, n_embd, 1, dh_max);
                ggml_tensor * a   = ggml_new_tensor_3d(sctx.get(), GGML_TYPE_F32, n_embd, n_expert_used, dh_max);
                ggml_tensor * db  = ggml_moe_doorbell(sctx.get(), ids, x, nullptr, 1, (int) dh_max, lane_frac);
                if (!ggml_backend_supports_op(devices[id].backend, db) ||
                    !ggml_backend_supports_op(devices[id].backend, ggml_moe_join(sctx.get(), a, db, nullptr, 1, (int) dh_max, 1))) {
                    LLAMA_LOG_WARN("%s: %s does not support the MoE CPU lane ops, LLAMA_MOE_CPU_LANE is off there\n", __func__, ggml_backend_name(devices[id].backend));
                    continue;
                }
            }
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
            typedef ggml_backend_buffer_t (*from_mapped_t)(ggml_backend_dev_t, void *, size_t);
            auto from_mapped = (from_mapped_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_buffer_from_mapped_host_ptr");
            const size_t size = chan_size*dd.rows.size();
            ggml_backend_buffer_t hb = host_buft && from_mapped ? ggml_backend_buft_alloc_buffer(host_buft, size) : nullptr;
            ggml_backend_buffer_t mb = hb ? from_mapped(dev, ggml_backend_buffer_get_base(hb), size) : nullptr;
            if (mb == nullptr) {
                if (hb) {
                    ggml_backend_buffer_free(hb);
                }
                LLAMA_LOG_WARN("%s: %s cannot map pinned host memory, LLAMA_MOE_CPU_LANE is off there\n", __func__, ggml_backend_name(devices[id].backend));
                continue;
            }
            ggml_backend_buffer_clear(hb, 0);
            lane_bufs.emplace_back(hb);
            lane_mapped.emplace_back(mb);
            for (size_t r = 0; r < dd.rows.size(); ++r) {
                layer & l = layers[dd.rows[r]];
                l.lane_host = (uint8_t *) ggml_backend_buffer_get_base(hb) + r*chan_size;
                l.lane_dev  = (uint8_t *) ggml_backend_buffer_get_base(mb) + r*chan_size;
                n_chan++;
            }
        }
        if (n_chan == 0) {
            lane_frac = 0.0f;
            return;
        }

        int n_threads = std::max(1, (int) std::thread::hardware_concurrency()/2 - 1);
        int poll = 50;
        if (const char * env = getenv("LLAMA_MOE_CPU_LANE_THREADS")) {
            n_threads = std::max(1, atoi(env));
        }
        if (const char * env = getenv("LLAMA_MOE_CPU_LANE_POLL")) {
            poll = atoi(env);
        }
        lane = std::make_unique<llama_moe_lane>(layers.size(), n_threads, poll);
        LLAMA_LOG_INFO("%s: MoE CPU lane: %zu layers, %.0f%% of the misses computed by %d CPU threads beside the device, "
            "channels %.2f MiB pinned (LLAMA_MOE_CPU_LANE=%.2f)\n", __func__, n_chan, 100.0*lane_frac, n_threads,
            chan_size*n_chan/1024.0/1024.0, lane_frac);
    }

    void * cpu_lane(int32_t il, int64_t n_tokens, int64_t n_used, const ggml_tensor * up, const ggml_tensor * gate,
            const ggml_tensor * down, const ggml_tensor * gate_up, float clamp) {
        if (!lane || !device_hits(il, n_tokens, n_used)) {
            return nullptr;
        }
        layer & l = layers[il];
        if (l.lane_dev == nullptr) {
            return nullptr;
        }
        if (!lane->is_set(il)) {
            const ggml_tensor * w_in = gate_up ? gate_up : up;
            bool ok = w_in != nullptr && down != nullptr && (gate_up != nullptr || gate != nullptr);
            for (const ggml_tensor * w : { up, gate, down, gate_up }) {
                ok = ok && (w == nullptr || (w->buffer && ggml_backend_buffer_is_host(w->buffer) && ggml_is_contiguous(w)));
            }
            if (!ok) {
                if (!l.lane_warned) {
                    LLAMA_LOG_WARN("%s: the experts of layer %d are not readable by the CPU, no CPU lane there\n", __func__, il);
                    l.lane_warned = true;
                }
                return nullptr;
            }
            llama_moe_lane_layer ll;
            ll.chan       = l.lane_host;
            ll.n_expert   = (int32_t) w_in->ne[2];
            ll.n_used     = (int32_t) n_used;
            ll.max_tokens = (int32_t) dh_max;
            ll.n_embd     = w_in->ne[0];
            ll.up         = up;
            ll.gate       = gate;
            ll.down       = down;
            ll.gate_up    = gate_up;
            ll.clamp      = clamp;
            lane->set(il, ll);
        }
        return l.lane_dev;
    }

    // LLAMA_MOE_DEVICE_HITS is set and not ruled out by the cache's settings (see init_device_hits)
    // the ring's borrow (large batches) and device hits (decode) take turns: borrow_slots commits the fills and updates the
    // device map, begin() returns the slots before a graph without the ring
    bool dh_requested() const {
        const char * env = getenv("LLAMA_MOE_DEVICE_HITS");
        return env != nullptr && atoi(env) > 0 && cpu_share <= 0.0f && (!borrow || ring > 0);
    }

    void init_device_hits() {
        if (cpu_share > 0.0f || (borrow && ring == 0)) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_DEVICE_HITS does not combine with --moe-cache-cpu-share or LLAMA_MOE_ASYNC_BORROW without LLAMA_MOE_RING, it is disabled\n", __func__);
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

            // LLAMA_MOE_DECODE_MEM: the slots are address space, backed from the lent memory during large batches (dm_switch)
            const bool lent_stage = !no_alloc && id < dm_devs.size() && dm_devs[id].lend > 0 && dm_devs[id].stage > 0;
            std::vector<size_t> start; // [expert tensor] offset of the staging slots in the banks of the lender
            if (!lent_stage && borrow && !plan_borrow(d, sec, start)) {
                LLAMA_LOG_WARN("%s: %s: the MoE cache is too small to lend %d staging slots, they are allocated separately\n", __func__, ggml_backend_buft_name(d.buft), n_stage);
            }

            if (lent_stage) {
                GGML_ASSERT(n_stage*d.slot_size <= dm_devs[id].stage);
                d.buf_stage.reset(vmm_alloc(d.buft, dm_devs[id].stage));
                if (!d.buf_stage) {
                    throw std::runtime_error("failed to reserve the address space of the MoE staging slots");
                }
            } else if (d.lender < 0) {
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
                    stage_view_layer[v] = il;
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
            if (ring > 0 && ring_prio) {
                typedef bool (*set_prio_t)(ggml_backend_t, int);
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                auto set_prio = reg ? (set_prio_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_stream_priority") : nullptr;
                if (set_prio && set_prio(d.copy.get(), 1)) {
                    LLAMA_LOG_INFO("%s: %10s MoE ring: the copy stream has the device's highest priority (LLAMA_MOE_RING_PRIO=1)\n", __func__,
                        ggml_backend_buft_name(d.buft));
                } else {
                    LLAMA_LOG_WARN("%s: %s: LLAMA_MOE_RING_PRIO=1 not applied (no stream priorities)\n", __func__, ggml_backend_name(d.backend));
                }
            }
            for (int32_t is = 0; is < n_stage; ++is) {
                d.ev_ready.emplace_back(ggml_backend_event_new(dev));
                if (!d.ev_ready.back()) {
                    throw std::runtime_error("failed to create the MoE staging stream");
                }
            }
            d.slot_layer.assign(n_stage, -1);
            if (ring > 0) {
                init_ring(d);
            }

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

    // LLAMA_MOE_RING: release events of the slots, the pinned identity map and zeros, the copy kernel
    void init_ring(device & d) {
        ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
        for (int32_t is = 0; is < n_stage; ++is) {
            d.ev_used.emplace_back(ggml_backend_event_new(dev));
            if (!d.ev_used.back()) {
                throw std::runtime_error("failed to create the MoE ring events");
            }
        }
        d.mtx = std::make_unique<std::mutex>();
        d.rs.init(n_stage);

        int64_t n_expert_max = 0;
        for (int32_t il : d.layers) {
            n_expert_max = std::max(n_expert_max, layers[il].stage_map->ne[1]);
        }
        d.pinned_map_size = n_expert_max*sizeof(int32_t);
        ggml_backend_buffer_type_t buft_pinned = ggml_backend_dev_host_buffer_type(dev);
        d.buf_pinned.reset(ggml_backend_buft_alloc_buffer(buft_pinned ? buft_pinned : ggml_backend_cpu_buffer_type(), d.pinned_map_size + zeros.size()));
        if (!d.buf_pinned) {
            throw std::runtime_error("failed to allocate the MoE ring buffers");
        }
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(d.buf_pinned.get());
        int32_t * map = (int32_t *) base;
        for (int64_t e = 0; e < n_expert_max; ++e) {
            map[e] = (int32_t) e;
        }
        memset(base + d.pinned_map_size, 0, zeros.size());
        d.pinned_map   = map;
        d.pinned_zeros = base + d.pinned_map_size;

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        copy_blocks = reg ? (copy_blocks_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_copy_blocks_async") : nullptr;
        if (copy_blocks == nullptr) {
            LLAMA_LOG_WARN("%s: %s has no block copy kernel, the MoE ring copies the cached experts one run at a time\n", __func__, ggml_backend_name(d.backend));
            return;
        }

        // the kernel reads the uncached experts in place from pinned host memory (one launch per 128 experts, not one copy per run)
        typedef ggml_backend_buffer_t (*from_mapped_t)(ggml_backend_dev_t, void *, size_t);
        auto from_mapped = ring_zc ? (from_mapped_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_buffer_from_mapped_host_ptr") : nullptr;
        std::unordered_map<ggml_backend_buffer_t, const uint8_t *> base_of_buf;
        size_t n_mapped = 0;
        for (int32_t il : d.layers) {
            layer & l = layers[il];
            l.ring_host.assign(l.experts.size(), nullptr);
            for (size_t ip = 0; ip < l.experts.size() && from_mapped; ++ip) {
                const ggml_tensor * w = l.experts[ip];
                ggml_backend_buffer_t hb = w->buffer;
                auto it = base_of_buf.find(hb);
                if (it == base_of_buf.end()) {
                    const uint8_t * base = nullptr;
                    if (ggml_backend_buffer_t mb = from_mapped(dev, ggml_backend_buffer_get_base(hb), ggml_backend_buffer_get_size(hb))) {
                        base = (const uint8_t *) ggml_backend_buffer_get_base(mb);
                        ggml_backend_buffer_free(mb); // the mapping is the pinned allocation's, the buffer only holds its address
                    }
                    it = base_of_buf.emplace(hb, base).first;
                }
                if (it->second != nullptr && ggml_is_contiguous(w)) {
                    l.ring_host[ip] = it->second + ((const uint8_t *) w->data - (const uint8_t *) ggml_backend_buffer_get_base(hb));
                    n_mapped++;
                }
            }
        }
        if (ring_zc && n_mapped == 0) {
            LLAMA_LOG_WARN("%s: %s cannot read the host experts in place (use --load-mode none), the MoE ring uploads them one run at a time\n",
                __func__, ggml_backend_name(d.backend));
        }
    }

    // place the staging slots of device id in the bank tails of the group plan_memory chose, slot is of tensor position ip at
    // start[ip] + is*sec[ip]; with LLAMA_MOE_DECODE_MEM they end below the group's lent tail, above the slots it keeps
    bool plan_borrow(device & d, const std::vector<size_t> & sec, std::vector<size_t> & start) {
        if (d.plan_lender < 0) {
            return false;
        }
        const group & g = groups[d.plan_lender];
        if (g.n_slots == 0 || g.ref.size() != sec.size()) {
            return false;
        }
        const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
        // with misses on the CPU, the n_zero slots after the cached ones must stay zero, so the staging slots end before them
        const bool lends = g.lend_first < g.n_slots;
        const int32_t end = lends ? g.lend_first : n_zero > 0 ? g.n_slots : g.n_slots + 1;
        const int32_t min_first = lends ? moe_cache_keep_slots(g.ref[0]->ne[2], n_used_batch(), transient_slots(g.n_slots)) : 0;
        for (int32_t k = moe_cache_stage_slots(bank_nb(g), sec, n_stage); k <= g.n_slots && end - k >= min_first; ++k) {
            const int32_t first = end - k;
            bool fits = true;
            start.assign(sec.size(), 0);
            for (size_t ip = 0; ip < sec.size(); ++ip) {
                start[ip] = GGML_PAD((size_t) first*g.banks[ip]->nb[2], alignment);
                fits = fits && start[ip] + n_stage*sec[ip] <= (size_t) end*g.banks[ip]->nb[2];
            }
            if (fits) {
                d.lender       = d.plan_lender;
                d.borrow_first = first;
                d.borrow_end   = std::min(end, g.n_slots);
                return true;
            }
        }
        return false;
    }

    // BEGIN ring borrow slots (extracted by the CPU harness)
    // take the lender's slots under the staging slots out of the SLRU, called on the compute stream before the first staging of a batch
    // the coldest experts of the lender leave the cache, the hotter ones under the staging slots move to their slots
    // with device hits (LLAMA_MOE_RING) the fills in flight are committed or dropped first and the device map follows the moves;
    // with submit threads the caller holds mtx_shared (dh_inflight, statistics), not the device's ring mutex
    void borrow_slots(device & d) {
        group & g = groups[d.lender];
        moe_cache_slru & c = g.slru;
        const int32_t first = d.borrow_first;
        const int32_t end   = d.borrow_end;
        const size_t  id    = (size_t) g.id;
        const bool    dh    = id < dh_devs.size() && !dh_devs[id].rows.empty();

        // the slot maps of earlier graphs are uploaded and no stream reads the banks any more
        ggml_backend_synchronize(d.copy.get());
        ggml_backend_synchronize(d.backend);
        if (dh && dh_devs[id].fill) {
            ggml_backend_synchronize(dh_devs[id].fill.get());
        }
        if (dh) {
            dm_flush_fills(id);
        }
        d.last_slot = -1;

        std::vector<bool> drop(g.n_slots, false);
        int32_t n_drop = 0;
        for (int seg = 0; seg < 2; ++seg) {
            for (int32_t s = c.head[seg]; s >= 0 && n_drop < end - first; s = c.next[s]) {
                drop[s] = true;
                n_drop++;
            }
        }
        GGML_ASSERT(n_drop == end - first);

        std::vector<int32_t> src;
        std::vector<int32_t> dst;
        for (int32_t s = first; s < end; ++s) {
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
        for (int32_t s = first; s < end; ++s) {
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

        // the device map after the moves, on the compute stream before the graph (its output rows may take the device path)
        if (dh) {
            dh_rebuild_map(dh_devs[id]);
            dh_devs[id].resync = false;
            ggml_backend_tensor_set_async(d.backend, dh_devs[id].map, dh_devs[id].map_host.data(), 0, ggml_nbytes(dh_devs[id].map));
            dh_devs[id].map_dirty = false;
        }
    }

    // give the borrowed slots back to the cache as free slots, before a batch that the cache serves fills it
    // the compute stream already waited for the last upload of the copy stream
    void return_slots(device & d) {
        moe_cache_slru & c = groups[d.lender].slru;
        for (int32_t s = d.borrow_end - 1; s >= d.borrow_first; --s) {
            GGML_ASSERT(c.key_of[s] < 0);
            c.push_front(s);
        }
        std::fill(d.slot_layer.begin(), d.slot_layer.end(), -1);
        d.borrowed = false;
    }
    // END ring borrow slots

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
        profile_save(true);
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
    // 0 tokens (the last layer of a prompt ubatch without outputs, cropped to the output rows) takes the cache path too: the
    // graph then has the topology of one with an output, which the worst case is reserved with, so it does not reallocate
    bool use_cache(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        return n_tokens >= (zero_rows ? 0 : 1) && n_tokens <= max_batch && std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) <= g.n_slots - g.tr.n;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        if (use_cache(il, n_tokens, n_expert_used)) {
            return l.slot_map;
        }
        if (n_tokens > 0 && l.stage_map != nullptr) {
            graph_stages = true;
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

            observe(il, ids, n);
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

    void dh_upload_map(size_t id) {
        dh_device & dd = dh_devs[id];
        if (dd.map_dirty) {
            ggml_backend_tensor_set_async(devices[id].backend, dd.map, dd.map_host.data(), 0, ggml_nbytes(dd.map));
            dd.map_dirty = false;
        }
        if (dh_mode == 2) {
            dh_queue(id);
        }
    }

    // dh_process with the adaptive policy, for all devices: hits and misses of the logged graph, then the best admissions
    void dh_process_adaptive() {
        for (auto & v : ad_cands_g) {
            v.clear();
        }
        int64_t n_tokens = 1;
        size_t n_miss_free = 0; // misses not already being filled
        for (size_t id = 0; id < dh_devs.size(); ++id) {
            dh_device & dd = dh_devs[id];
            for (size_t r = 0; r < dd.rows.size(); ++r) {
                const int32_t il = dd.rows[r];
                const layer & l = layers[il];
                moe_cache_slru & slru = groups[l.ig].slru;
                const int32_t * ids = dd.log_host.data() + r*dh_log_cap;
                int64_t n = 0;
                while (n < dh_log_cap && ids[n] >= 0) {
                    n++;
                }
                if (n == 0) {
                    continue;
                }
                n_tokens = std::max<int64_t>(n_tokens, n / n_expert_used);
                size_t expert_bytes = 0;
                for (const ggml_tensor * w : l.experts) {
                    expert_bytes += w->nb[2];
                }
                observe(il, ids, n);
                slru.mark_seen(ids, n);
                const int32_t * slots = slru.slot_map[il];
                for (int32_t e : slru.uniq) {
                    if (slots[e] < slru.n_slots) {
                        slru.touch(slots[e]);
                        stats_dh.hits++;
                    } else {
                        stats_dh.misses++;
                        stats_dh.bytes_miss += expert_bytes;
                        if (dh_inflight.count(il*slru.n_expert + e) == 0) {
                            ad_cands_g[l.ig].push_back({ il, e });
                            n_miss_free++;
                        }
                    }
                }
            }
        }

        ad_swaps.clear();
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            if (ad_cands_g[ig].empty()) {
                continue;
            }
            ad_swaps_g.clear();
            ad.pair(groups[ig].slru, ad_cands_g[ig], -1, ad_swaps_g, ad_victims);
            ad_swaps.insert(ad_swaps.end(), ad_swaps_g.begin(), ad_swaps_g.end());
        }
        std::sort(ad_swaps.begin(), ad_swaps.end(), [](const moe_cache_policy::swap & a, const moe_cache_policy::swap & b) {
            return a.gain != b.gain ? a.gain > b.gain : (a.il != b.il ? a.il < b.il : a.expert < b.expert);
        });
        const size_t budget = (size_t) ad_fills_per_token*(size_t) n_tokens;
        std::vector<int32_t> per_dev(dh_devs.size(), 0);
        size_t n_adm = 0;
        for (const auto & sw : ad_swaps) {
            const group & g = groups[layers[sw.il].ig];
            if (n_adm >= budget || per_dev[g.id] >= dh_fill_max) {
                stats_ad.capped++;
                continue;
            }
            moe_cache_slru & slru = groups[layers[sw.il].ig].slru;
            const int32_t s   = sw.slot;
            const int32_t key = slru.key_of[s];
            if (key >= 0) {
                slru.slot_map[key / slru.n_expert][key % slru.n_expert] = slru.empty;
                dh_map_set(key / slru.n_expert, key % slru.n_expert, -1 - key % slru.n_expert);
                slru.n_filled--;
            } else {
                stats_ad.into_free++;
            }
            slru.key_of[s] = -1;
            slru.unlink(s);
            dh_devs[g.id].pending.push_back({ sw.il, sw.expert, s });
            dh_inflight.insert(sw.il*slru.n_expert + sw.expert);
            per_dev[g.id]++;
            n_adm++;
        }
        stats_ad.admitted += n_adm;
        stats_ad.kept_out += n_miss_free - n_adm;
    }


    // LLAMA_MOE_DECODE_MEM runtime

    ggml_backend_buffer_type_t compute_buft(ggml_backend_t backend) const {
        for (size_t id = 0; id < devices.size(); ++id) {
            if (devices[id].backend == backend && id < dm_devs.size() && dm_devs[id].lend > 0) {
                return dm_devs[id].cbuft;
            }
        }
        return nullptr;
    }

    // after each graph allocation, before its tensors are written: back the compute memory it uses; a device whose graph
    // needs more than the small part takes the memory from its bank tails, and gives it back after small graphs again
    void on_alloc(ggml_backend_sched_t sched) {
        const bool stages = graph_stages;
        graph_stages = false;
        if (!dm_on) {
            return;
        }
        std::vector<size_t> take;
        std::vector<size_t> give;
        std::vector<size_t> need(devices.size(), 0);
        for (size_t id = 0; id < devices.size(); ++id) {
            dm_device & dm = dm_devs[id];
            if (dm.lend == 0) {
                continue;
            }
            ggml_backend_buffer_t buf = nullptr;
            size_t used = ggml_backend_sched_get_buffer_used(sched, devices[id].backend, &buf);
            if (buf != dm.cbuf) {
                // a new compute buffer (reserve or reallocation): nothing of it is backed
                dm.cbuf       = buf;
                dm.committed  = 0;
                dm.alias_comp = 0;
                if (buf != nullptr && dm_alias) {
                    dm_alias_compute(id);
                }
            }
            if (buf == nullptr) {
                continue;
            }
            const size_t bsize = ggml_backend_buffer_get_size(buf);
            used = std::min(used, bsize);
            need[id] = used;
            // a graph with staged layers writes the staging slots, which are backed only while lent
            if (used <= dm.small && !(stages && dm.stage > 0)) {
                dm.used_small_max = std::max(dm.used_small_max, used);
                if (dm.lent) {
                    if (++dm.n_small >= dm_return_after) {
                        give.push_back(id);
                    }
                } else {
                    if (dm.committed < std::min(dm.small, bsize)) {
                        dm_commit_compute(id, std::min(dm.small, bsize));
                    }
                    if (++dm.n_retry % 64 == 0) {
                        bool out = false;
                        for (const group & g : groups) {
                            out = out || (g.id == (int32_t) id && g.tail_out);
                        }
                        if (out) {
                            dm_retry_tails(id); // the VMM calls synchronize the device; the tails are out of every map
                        }
                    }
                }
            } else {
                dm.used_large_max = std::max(dm.used_large_max, used);
                dm.n_small = 0;
                if (!dm.lent) {
                    if (used > dm.small && used <= dm.large/4 && !dm_warned) {
                        // a graph this far below the worst case is likely a decode graph: each one would switch
                        LLAMA_LOG_WARN("%s: %s: a graph uses %.2f MiB of compute buffer, more than the %.2f MiB kept for small batches: "
                            "raise LLAMA_MOE_DECODE_MEM_SPARE_MIB or LLAMA_MOE_DECODE_MEM_TOKENS\n", __func__,
                            ggml_backend_name(devices[id].backend), used/1024.0/1024.0, dm.small/1024.0/1024.0);
                        dm_warned = true;
                    }
                    take.push_back(id);
                } else if (used > dm.committed) {
                    dm_commit_compute(id, used);
                }
            }
        }
        if (!take.empty() || !give.empty()) {
            dm_switch(sched, take, give, need);
        }
    }

    void dm_commit_compute(size_t id, size_t end) {
        dm_device & dm = dm_devs[id];
        if (end <= dm.committed) {
            return;
        }
        if (!dm_commit(dm.cbuf, dm.committed, end - dm.committed)) {
            GGML_ABORT("%s: cannot back %.2f MiB of the compute buffer of %s (LLAMA_MOE_DECODE_MEM): a smaller --moe-cache-mib or "
                "LLAMA_MOE_DECODE_MEM_SPARE_MIB leaves room", __func__, (end - dm.committed)/1024.0/1024.0, ggml_backend_name(devices[id].backend));
        }
        dm.committed = end;
    }

    // nothing in flight on any device or stream of the cache
    void dm_sync_all(ggml_backend_sched_t sched) {
        ggml_backend_sched_synchronize(sched);
        for (size_t id = 0; id < devices.size(); ++id) {
            if (devices[id].copy) {
                ggml_backend_synchronize(devices[id].copy.get());
            }
            if (id < dh_devs.size() && dh_devs[id].fill) {
                ggml_backend_synchronize(dh_devs[id].fill.get());
            }
        }
    }

    // BEGIN decode_mem slots (extracted by the CPU harness)
    // after dm_sync_all: the queued fills are done (committed), the pending ones give their slots back
    void dm_flush_fills(size_t id) {
        if (id >= dh_devs.size()) {
            return;
        }
        dh_device & dd = dh_devs[id];
        for (const dh_fill & f : dd.queued) {
            dh_commit(f);
        }
        dd.queued.clear();
        for (const dh_fill & f : dd.pending) {
            moe_cache_slru & c = groups[layers[f.il].ig].slru;
            dh_inflight.erase(f.il*c.n_expert + f.expert);
            c.push_front(f.slot);
        }
        dd.pending.clear();
    }

    // the tail [lend_first, n_slots) leaves the cache: the coldest experts of the group are evicted, the hotter ones in the tail
    // move to the freed slots below it (D2D on the compute stream); every slot of the group is in the SLRU lists here, except
    // the transient ones at its start
    void dm_evict_tail(group & g) {
        moe_cache_slru & c = g.slru;
        const int32_t first = g.lend_first;
        const int32_t k     = g.n_slots - first;
        if (k <= 0) {
            return;
        }
        std::vector<bool> drop(g.n_slots, false);
        int32_t n_drop = 0;
        for (int seg = 0; seg < 2 && n_drop < k; ++seg) {
            for (int32_t s = c.head[seg]; s >= 0 && n_drop < k; s = c.next[s]) {
                drop[s] = true;
                n_drop++;
            }
        }
        GGML_ASSERT(n_drop == k);

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
                c.n_filled--;
                stats_dm.evicted++;
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
                continue;
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

        ggml_backend_t backend = devices[g.id].backend;
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
                ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
            }
            i += n;
        }
        stats_dm.moved += n_move;
    }

    // refill order: the decayed routing counts (adaptive policy) or this run's routings first, then the profile
    std::pair<double, double> dm_score(int32_t il, int32_t e) const {
        const size_t key = (size_t) il*n_expert_model + e;
        double primary = 0.0;
        if (ad_on) {
            primary = ad.value((int32_t) key);
        } else if (!prof_run.empty()) {
            primary = prof_run[key];
        }
        const double secondary = prof.n_layer == (int32_t) layers.size() ? prof.get(il, e) : 0.0;
        return { primary, secondary };
    }

    // the returned tail: free slots, the next victims, and the hottest experts missing to put there
    void dm_return_tail(group & g) {
        moe_cache_slru & c = g.slru;
        for (int32_t s = g.n_slots - 1; s >= g.lend_first; --s) {
            GGML_ASSERT(c.key_of[s] < 0);
            c.push_front(s);
        }
        const size_t k = (size_t) (g.n_slots - g.lend_first);
        std::vector<std::pair<std::pair<double, double>, std::pair<int32_t, int32_t>>> cand;
        for (int32_t il : g.layers) {
            for (int32_t e = 0; e < c.n_expert; ++e) {
                if (c.slot_map[il][e] < c.n_slots || dh_inflight.count(il*c.n_expert + e) > 0) {
                    continue;
                }
                const auto sc = dm_score(il, e);
                if (sc.first > 0.0 || sc.second > 0.0) {
                    cand.push_back({ sc, { il, e } });
                }
            }
        }
        auto hotter = [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; };
        if (cand.size() > k) {
            std::partial_sort(cand.begin(), cand.begin() + k, cand.end(), hotter);
            cand.resize(k);
        } else {
            std::sort(cand.begin(), cand.end(), hotter);
        }
        g.refill.clear();
        for (const auto & ce : cand) {
            g.refill.push_back(ce.second);
        }
        g.refill_pos = 0;
    }

    // up to dm_refill_cap of the refills of the device's groups: on the fill stream with device hits (committed by the next
    // graph like a miss fill), else on the compute stream before this graph
    void dm_refill(size_t id) {
        if (!dm_on || id >= dm_devs.size() || dm_devs[id].lend == 0 || dm_devs[id].lent) {
            return;
        }
        const bool dh = id < dh_devs.size() && !dh_devs[id].rows.empty();
        int64_t budget = dm_refill_cap > 0 ? dm_refill_cap : INT64_MAX;
        bool any = false;
        for (group & g : groups) {
            if (g.id != (int32_t) id) {
                continue;
            }
            moe_cache_slru & c = g.slru;
            while (budget > 0 && g.refill_pos < g.refill.size()) {
                const int32_t il  = g.refill[g.refill_pos].first;
                const int32_t e   = g.refill[g.refill_pos].second;
                const int32_t key = il*c.n_expert + e;
                if (c.slot_map[il][e] < c.n_slots || dh_inflight.count(key) > 0) {
                    g.refill_pos++;
                    continue;
                }
                const int32_t s = c.head[0];
                if (s < 0 || c.key_of[s] >= 0) {
                    g.refill_pos = g.refill.size(); // no free slot left: the misses took them
                    break;
                }
                g.refill_pos++;
                const layer & l = layers[il];
                size_t bytes = 0;
                if (dh) {
                    c.unlink(s);
                    dh_devs[id].pending.push_back({ il, e, s });
                    dh_inflight.insert(key);
                    for (const ggml_tensor * w : l.experts) {
                        bytes += w->nb[2];
                    }
                } else {
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const ggml_tensor * w = l.experts[ip];
                        ggml_backend_tensor_set_async(devices[id].backend, g.banks[ip], (const uint8_t *) w->data + e*w->nb[2], s*w->nb[2], w->nb[2]);
                        bytes += w->nb[2];
                    }
                    c.key_of[s] = key;
                    c.slot_map[il][e] = s;
                    c.move(s, false);
                    c.n_filled++;
                }
                stats_dm.refilled++;
                stats_dm.refill_bytes += bytes;
                budget--;
                any = true;
            }
        }
        if (any) {
            stats_dm.refill_graphs++;
        }
    }
    // END decode_mem slots

    // memory moves between buffers of a device: through the spare granules when the backend has them (no cuMemCreate/Release,
    // no zeroing); the spares left over after a give are freed by dm_trim
    size_t dm_release(ggml_backend_buffer_t buf, size_t off, size_t size) {
        return vmm_release_spare ? vmm_release_spare(buf, off, size) : vmm_release(buf, off, size);
    }
    bool dm_commit(ggml_backend_buffer_t buf, size_t off, size_t size) {
        return vmm_commit_spare ? vmm_commit_spare(buf, off, size, false) : vmm_commit(buf, off, size);
    }
    void dm_trim(const device & d) {
        if (vmm_trim) {
            vmm_trim(d.buft);
        }
    }

    // back a group's tail again; false (nothing backed) if the device has no memory for it
    bool dm_back_tail(device & d, const group & g) {
        for (size_t i = 0; i < g.lend_ranges.size(); ++i) {
            if (!dm_commit(d.buf.get(), g.lend_ranges[i].first, g.lend_ranges[i].second)) {
                for (size_t j = 0; j < i; ++j) {
                    dm_release(d.buf.get(), g.lend_ranges[j].first, g.lend_ranges[j].second);
                }
                return false;
            }
        }
        return true;
    }

    // a tail that stayed out after a give (another context took the memory meanwhile, e.g. an MTP draft context) is backed
    // again once there is room, tried every 64 small graphs
    void dm_retry_tails(size_t id) {
        device & d = devices[id];
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(d.backend), &free, &total);
        bool any = false;
        for (group & g : groups) {
            if (g.id == (int32_t) id && g.tail_out) {
                size_t bytes = 0;
                for (const auto & r : g.lend_ranges) {
                    bytes += r.second;
                }
                if (bytes > free || !dm_back_tail(d, g)) {
                    break;
                }
                free -= bytes;
                g.tail_out = false;
                dm_return_tail(g);
                any = true;
            }
        }
        dm_trim(d);
        if (any) {
            LLAMA_LOG_INFO("%s: %s: the MoE cache tails that stayed out are backed again\n", __func__, ggml_backend_name(d.backend));
            if (id < dh_devs.size() && !dh_devs[id].rows.empty()) {
                dh_devs[id].resync = true;
            }
        }
    }

    void dm_switch(ggml_backend_sched_t sched, const std::vector<size_t> & take, const std::vector<size_t> & give, const std::vector<size_t> & need) {
        const int64_t t0 = ggml_time_us();
        dm_sync_all(sched);
        const int64_t t_sync = ggml_time_us();

        for (size_t id : take) {
            dm_flush_fills(id);
            for (group & g : groups) {
                if (g.id == (int32_t) id && !g.tail_out) {
                    dm_evict_tail(g);
                }
            }
            if (id < dh_devs.size() && !dh_devs[id].rows.empty()) {
                dh_devs[id].resync = true;
            }
        }
        const int64_t t_evict = ggml_time_us();

        // the memory moves between the banks and the compute buffer, one thread per device (the VMM calls synchronize the device;
        // the moves of dm_evict_tail are queued on the compute stream and done by the first of them)
        std::vector<std::string> err(devices.size());
        std::vector<size_t> released(devices.size(), 0);
        std::vector<int64_t> us_tail(devices.size(), 0);
        std::vector<int64_t> us_comp(devices.size(), 0);
        std::vector<int64_t> us_stage(devices.size(), 0);
        std::vector<std::vector<size_t>> out(devices.size()); // groups whose tail stays out
        auto work = [&](size_t id, bool is_take) {
            device & d = devices[id];
            dm_device & dm = dm_devs[id];
            int64_t t = ggml_time_us();
            auto lap = [&t](int64_t & acc) {
                const int64_t now = ggml_time_us();
                acc += now - t;
                t = now;
            };
            if (is_take) {
                if (dm_alias) {
                    // the tails stay mapped (also as compute and staging memory): the moves out of them finish first
                    ggml_backend_synchronize(d.backend);
                    released[id] += dm_tail_bytes(id);
                } else {
                    for (const group & g : groups) {
                        if (g.id == (int32_t) id) {
                            for (const auto & r : g.lend_ranges) {
                                released[id] += dm_release(d.buf.get(), r.first, r.second);
                            }
                        }
                    }
                }
                lap(us_tail[id]);
                if (need[id] > dm.committed && !dm_commit(dm.cbuf, dm.committed, need[id] - dm.committed)) {
                    err[id] = "cannot back the compute buffer of a large batch";
                    return;
                }
                dm.committed = std::max(dm.committed, need[id]);
                lap(us_comp[id]);
                if (dm.stage > dm.alias_stage && d.buf_stage) {
                    // the ring writes every expert of a slot before a matmul reads it; past the last one the MMQ reads rows it
                    // does not keep, so the slots need no zeroing
                    if (!dm_commit(d.buf_stage.get(), dm.alias_stage, dm.stage - dm.alias_stage)) {
                        err[id] = "cannot back the staging slots of a large batch";
                        return;
                    }
                }
                lap(us_stage[id]);
                dm.lent = true;
            } else {
                // aliased: only what a large batch needed beyond the tail space goes back
                const size_t keep = std::min(dm.small, ggml_backend_buffer_get_size(dm.cbuf)) + dm.alias_comp;
                dm_release(dm.cbuf, keep, SIZE_MAX/4); // up to the end of the address space
                dm.committed = std::min(dm.committed, keep);
                lap(us_comp[id]);
                if (dm.stage > dm.alias_stage && d.buf_stage) {
                    dm_release(d.buf_stage.get(), dm.alias_stage, dm.stage - dm.alias_stage);
                }
                lap(us_stage[id]);
                if (need[id] > dm.committed && !dm_commit(dm.cbuf, dm.committed, need[id] - dm.committed)) {
                    err[id] = "cannot back the compute buffer of a small batch";
                    return;
                }
                dm.committed = std::max(dm.committed, need[id]);
                // a tail without memory (another context took it while lent) stays out of the cache, retried later
                for (size_t ig = 0; ig < groups.size() && !dm_alias; ++ig) {
                    group & g = groups[ig];
                    if (g.id == (int32_t) id && !g.tail_out && !dm_back_tail(d, g)) {
                        out[id].push_back(ig);
                    }
                }
                lap(us_tail[id]);
                dm.lent    = false;
                dm.n_small = 0;
                // what the tails do not take back goes to the driver; while lent the spares stay ours (the plan counts
                // them for the cache, a context that took them would leave the tails without memory at the give)
                dm_trim(d);
            }
        };
        std::vector<std::pair<size_t, bool>> jobs;
        for (size_t id : take) {
            jobs.push_back({ id, true });
        }
        for (size_t id : give) {
            jobs.push_back({ id, false });
        }
        if (jobs.size() == 1) {
            work(jobs[0].first, jobs[0].second);
        } else {
            std::vector<std::thread> threads;
            for (const auto & j : jobs) {
                threads.emplace_back(work, j.first, j.second);
            }
            for (auto & t : threads) {
                t.join();
            }
        }
        const int64_t t_vmm = ggml_time_us();
        for (size_t id = 0; id < devices.size(); ++id) {
            if (!err[id].empty()) {
                GGML_ABORT("%s: %s: %s (LLAMA_MOE_DECODE_MEM; a smaller --moe-cache-mib or a larger LLAMA_MOE_CACHE_MARGIN_MIB leaves room)",
                    __func__, ggml_backend_name(devices[id].backend), err[id].c_str());
            }
            stats_dm.released += released[id];
            for (size_t ig : out[id]) {
                groups[ig].tail_out   = true;
                groups[ig].refill_pos = groups[ig].refill.size();
                stats_dm.tails_out++;
                LLAMA_LOG_WARN("%s: %s: no memory to back %d slots of the MoE cache again after a large batch (another context took it "
                    "meanwhile): they stay out of the cache and are retried; a larger LLAMA_MOE_CACHE_MARGIN_MIB keeps room\n", __func__,
                    ggml_backend_name(devices[id].backend), groups[ig].n_slots - groups[ig].lend_first);
            }
        }

        for (size_t id : give) {
            for (group & g : groups) {
                if (g.id == (int32_t) id && !g.tail_out) {
                    dm_return_tail(g);
                }
            }
        }
        // staging slots backed from the lent memory hold nothing after a take or a give
        for (const auto & j : jobs) {
            device & d = devices[j.first];
            if (dm_devs[j.first].stage > 0 && !d.slot_layer.empty()) {
                std::fill(d.slot_layer.begin(), d.slot_layer.end(), -1);
                d.last_slot  = -1;
                d.ring_fence = true;
            }
        }

        // the device map now, not at the next begin(): with submit threads the chunks of a batch have no begin(), and its
        // output rows may take the device path (the old map points into the released tails)
        for (size_t id : take) {
            if (id < dh_devs.size() && !dh_devs[id].rows.empty()) {
                dh_device & dd = dh_devs[id];
                dh_rebuild_map(dd);
                dd.resync = false;
                ggml_backend_tensor_set_async(devices[id].backend, dd.map, dd.map_host.data(), 0, ggml_nbytes(dd.map));
                dd.map_dirty = false;
            }
        }

        // cost model: host sync of every stream, the evictions (host side; their D2D moves run before the first VMM call),
        // then per device (slowest one) the tails, the compute buffer and the staging slots
        const int64_t t1 = ggml_time_us();
        int64_t m_tail = 0, m_comp = 0, m_stage = 0;
        for (size_t id = 0; id < devices.size(); ++id) {
            m_tail  = std::max(m_tail,  us_tail[id]);
            m_comp  = std::max(m_comp,  us_comp[id]);
            m_stage = std::max(m_stage, us_stage[id]);
        }
        dm_cost & c = take.empty() ? stats_dm.give_cost : stats_dm.take_cost;
        c.sync  += t_sync - t0;
        c.evict += t_evict - t_sync;
        c.tail  += m_tail;
        c.comp  += m_comp;
        c.stage += m_stage;
        c.rest  += t1 - t_vmm;
        const int64_t dt = t1 - t0;
        const size_t n_sw = take.empty() ? stats_dm.gives + 1 : stats_dm.takes + 1;
        if (n_sw <= 2) {
            LLAMA_LOG_INFO("%s: %s %zu: %zu devices in %.1f ms = sync %.1f + evict %.1f + VMM %.1f (slowest device: tails %.1f, compute %.1f, "
                "staging %.1f) + rest %.1f ms\n", __func__, take.empty() ? "give" : "take", n_sw, jobs.size(), dt/1000.0,
                (t_sync - t0)/1000.0, (t_evict - t_sync)/1000.0, (t_vmm - t_evict)/1000.0, m_tail/1000.0, m_comp/1000.0,
                m_stage/1000.0, (t1 - t_vmm)/1000.0);
        }
        if (!take.empty()) {
            stats_dm.takes++;
            stats_dm.take_us += dt;
        } else {
            stats_dm.gives++;
            stats_dm.give_us += dt;
        }
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
            if (dh_logged && !ad_on) {
                dh_process(id);
            }
            if (!ad_on) {
                dm_refill(id);
                dh_upload_map(id);
            }
        }
        if (ad_on) {
            // the policy ranks the candidates of all devices against one per-token cap
            if (dh_logged) {
                dh_process_adaptive();
            }
            for (size_t id = 0; id < dh_devs.size(); ++id) {
                if (!dh_devs[id].rows.empty()) {
                    dm_refill(id);
                    dh_upload_map(id);
                }
            }
        }
        if (dh_logged) {
            stats_dh.graphs++;
        }
        dh_logged = logs;
    }

    void launched() {
        ring_end();
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
        ggml_context * ctx_v = ctx_views_dev[g.id].get();
        ggml_reset(ctx_v);
        ggml_tensor * src_view = ggml_view_3d(ctx_v, bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_v, dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        __atomic_fetch_add(&stats_copy.hits,  (size_t) n, __ATOMIC_RELAXED);
        __atomic_fetch_add(&stats_copy.bytes, ggml_nbytes(src_view), __ATOMIC_RELAXED);

        return n;
    }

    const void * cached_expert(ggml_backend_t backend, const ggml_tensor * w, int64_t e, size_t size) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return nullptr;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return nullptr;
        }
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return nullptr;
        }
        const ggml_tensor * bank = g.banks[b.ip];
        GGML_ASSERT(bank->nb[1] == w->nb[1] && size <= bank->nb[2]);

        __atomic_fetch_add(&stats_copy.hits,  (size_t) 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&stats_copy.bytes, size, __ATOMIC_RELAXED);

        return (const uint8_t *) bank->data + (size_t) slots[e]*bank->nb[2];
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
        std::lock_guard<std::mutex> lock(mtx_shared);
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
        if (ad_on) {
            ad.tick++;
            for (group & g : groups) {
                clear_transient(g);
            }
        }
        if (!prof_save.empty() && prof_save_min > 0.0 && (++prof_graphs & 63) == 0 &&
                ggml_time_us() - prof_saved_us >= (int64_t) (prof_save_min*60e6)) {
            profile_save(false);
        }
        auto plans = ring_plan(gf);
        ring_return_idle(plans);
        dh_begin(gf);
        for (size_t id = 0; id < devices.size() && dm_on; ++id) {
            if (id >= dh_devs.size() || dh_devs[id].rows.empty()) {
                dm_refill(id);
            }
        }
        ring_begin(plans);
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

        if (!stash.empty()) {
            stash.clear();
        }
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

    // LLAMA_MOE_RING (Strata's prompt expert stream): before a graph runs, the staged layers of each device are planned in graph
    // order and the device's copy stream starts filling the K staging slots with every expert of the first K layers that is not in
    // the cache (host -> device copies) and the cached ones (one copy kernel per expert tensor), independent of the routing. At
    // each staged layer the compute stream releases the slot of the layer before (an event after its matmuls), the copy stream
    // takes the next planned layer into it, and the compute stream waits only for its own layer's slot: no host step per layer,
    // the copies run on the copy engine up to K-1 layers ahead. Partial layers (batches under the full-batch rule) read their
    // routing back and are staged at their own layer.
    // ring_begin_device / ring_end_device record on the device's compute stream: with a submit thread per device they belong on
    // that thread, around its part of the graph.
    std::vector<std::vector<llama_moe_ring_sched::entry>> ring_plan(ggml_cgraph * gf) const {
        std::vector<std::vector<llama_moe_ring_sched::entry>> plans(devices.size());
        if (ring == 0 || gf == nullptr) {
            return plans;
        }
        // the matmuls that read a staging slot: the scheduler has already replaced the staging maps in the graph by their per-split
        // input copies (ggml_backend_sched_alloc_graph runs before the compute), the device weights they read stay as they are
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            const auto it = stage_view_layer.find(node->src[0]);
            if (it == stage_view_layer.end()) {
                continue;
            }
            const layer & l = layers[it->second];
            const int64_t n_ids = ggml_nelements(node->src[2]);
            auto & plan = plans[groups[l.ig].id];
            if (n_ids == 0 || (!plan.empty() && plan.back().il == it->second)) {
                continue; // gate, up and down of one layer
            }
            llama_moe_ring_sched::entry e;
            e.il   = it->second;
            e.slot = l.rank % n_stage;
            e.full = n_ids >= full_ids*l.stage_map->ne[1];
            plan.push_back(e);
        }
        return plans;
    }

    void ring_begin(std::vector<std::vector<llama_moe_ring_sched::entry>> & plans) {
        for (size_t id = 0; id < devices.size() && ring > 0; ++id) {
            if (devices[id].mtx) {
                ring_begin_device(devices[id], std::move(plans[id]));
            }
        }
    }

    // a graph that does not stage on a device gives the borrowed slots back before anything fills the cache (with device hits
    // no lookup of the cache path would do it); the compute stream waits for the ring's last copy into them
    void ring_return_idle(const std::vector<std::vector<llama_moe_ring_sched::entry>> & plans) {
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (!d.mtx || !d.borrowed || !plans[id].empty()) {
                continue;
            }
            std::lock_guard<std::mutex> lock(*d.mtx);
            if (d.last_slot >= 0) {
                ggml_backend_event_wait(d.backend, d.ev_ready[d.last_slot].get());
                d.last_slot = -1;
            }
            d.rs.drop();
            d.ring_prev = false;
            return_slots(d);
        }
    }

    // submit threads: the plan of the graph in pipeline copy k, started by the device's own thread (submit_graph)
    void ring_plan_copy(ggml_cgraph * gf, int k) {
        if (ring == 0) {
            return;
        }
        auto plans = ring_plan(gf);
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (d.mtx) {
                std::lock_guard<std::mutex> lock(*d.mtx);
                if ((int) d.ring_plans.size() <= k) {
                    d.ring_plans.resize(k + 1);
                }
                d.ring_plans[k] = std::move(plans[id]);
            }
        }
    }

    void submit_graph(ggml_backend_t backend, int k, bool begin) {
        if (ring == 0) {
            return;
        }
        for (device & d : devices) {
            if (d.backend != backend || !d.mtx) {
                continue;
            }
            if (!begin) {
                ring_end_device(d);
                return;
            }
            std::vector<llama_moe_ring_sched::entry> plan;
            {
                std::lock_guard<std::mutex> lock(*d.mtx);
                if (k >= 0 && k < (int) d.ring_plans.size()) {
                    plan.swap(d.ring_plans[k]);
                }
            }
            ring_begin_device(d, std::move(plan));
            return;
        }
    }

    // before a batch queued by submit threads: what begin() and launched() do between graphs, once, on the calling thread
    // (the threads skip both); the device hits' log of the last graph is processed and its fills queued, and the log of
    // the batch's last graph is read by the next begin()
    void submit_settle() {
        if (dh_devs.empty()) {
            return;
        }
        dh_begin(nullptr);
        if (dh_mode == 1) {
            for (size_t id = 0; id < dh_devs.size(); ++id) {
                if (!dh_devs[id].rows.empty()) {
                    dh_queue(id);
                }
            }
        }
        dh_logged = true;
    }

    void ring_begin_device(device & d, std::vector<llama_moe_ring_sched::entry> plan) {
        // the borrow changes shared state (dh_inflight, statistics): mtx_shared first, the order of copy()
        std::unique_lock<std::mutex> lock_shared(mtx_shared, std::defer_lock);
        if (!plan.empty() && d.lender >= 0 && !d.borrowed) {
            lock_shared.lock();
        }
        std::lock_guard<std::mutex> lock(*d.mtx);
        if (plan.empty()) {
            d.rs.drop();
            d.ring_prev = false;
            return;
        }
        if (d.lender >= 0 && !d.borrowed) {
            borrow_slots(d);
            d.ring_fence = true;
        }
        // after a graph without the ring (fills, borrowed slots moved, per-layer staging) the copy stream waits for all of it once;
        // after a ring graph each slot waits for the release of its last layer, so the next graph's first layers can arrive while
        // the last ones of this graph still compute
        const bool fence = !d.ring_prev || d.ring_fence;
        if (fence) {
            ggml_backend_event_record(d.ev_compute.get(), d.backend);
            ggml_backend_event_wait(d.copy.get(), d.ev_compute.get());
        }
        d.rs.start(std::move(plan), fence);
        d.ring_prev  = true;
        d.ring_fence = false;
        if (ring_log) {
            d.rstats_graph = d.rstats;
            d.rgraph_t0    = ggml_time_us();
        }
        d.rstats.graphs++;
        ring_issue(d);
    }

    // the graph is queued: the slot of the device's last staged layer is released after it
    void ring_end() {
        if (ring == 0) {
            return;
        }
        for (device & d : devices) {
            if (d.mtx) {
                ring_end_device(d);
            }
        }
    }

    void ring_end_device(device & d) {
        std::lock_guard<std::mutex> lock(*d.mtx);
        const int32_t rel = d.rs.release();
        if (rel >= 0) {
            ggml_backend_event_record(d.ev_used[rel].get(), d.backend);
        }
        if (ring_log && d.rgraph_t0 > 0) {
            const auto & a = d.rstats_graph;
            const auto & b = d.rstats;
            LLAMA_LOG_INFO("llama_moe_cache: ring %s graph %zu: %zu layers ahead, %zu at their layer, %zu partial, %zu dropped; %.2f GiB "
                "from host, %.2f GiB from the cache; host %.1f ms queueing copies, %.1f ms queueing the graph\n",
                ggml_backend_name(d.backend), b.graphs, b.layers - a.layers, b.at_layer - a.at_layer, b.partial - a.partial,
                b.dropped - a.dropped, (b.bytes_host - a.bytes_host)/1024.0/1024.0/1024.0, (b.bytes_cache - a.bytes_cache)/1024.0/1024.0/1024.0,
                (b.issue_us - a.issue_us)/1000.0, (ggml_time_us() - d.rgraph_t0)/1000.0);
            d.rgraph_t0 = 0;
        }
    }

    // the planned full layers whose slot is free, in order
    void ring_issue(device & d) {
        for (int64_t i = d.rs.can_issue(); i >= 0; i = d.rs.can_issue()) {
            ring_fill(d, (size_t) i, nullptr);
            d.rstats.layers++;
        }
    }

    void ring_fill(device & d, size_t i, const std::vector<bool> * used) {
        const llama_moe_ring_sched::entry e = d.rs.plan[i];
        GGML_ASSERT(!e.issued && d.rs.holder[e.slot] < 0);
        if (d.rs.issue(i)) {
            ggml_backend_event_wait(d.copy.get(), d.ev_used[e.slot].get());
        }
        ring_stage(d, e.il, e.slot, used);
    }

    // the copies of layer il into slot is on the copy stream: uncached experts from host memory, cached ones from the banks
    void ring_stage(device & d, int32_t il, int32_t is, const std::vector<bool> * used) {
        const int64_t t0 = ggml_time_us();
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        const int32_t * slots = g.slru.slot_map[il];
        ggml_backend_t copy = d.copy.get();
        const int32_t n_expert = (int32_t) l.stage_map->ne[1];

        llama_moe_ring_copies & cp = d.r_cp;
        const bool zc = copy_blocks != nullptr && !l.ring_host.empty() &&
            std::all_of(l.ring_host.begin(), l.ring_host.end(), [](const uint8_t * p) { return p != nullptr; });
        cp.plan(slots, g.n_slots, n_expert, used, zc ? 0 : ring_merge);

        for (size_t ip = 0; ip < l.experts.size() && zc; ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * st   = g.stage[is][ip];
            const ggml_tensor * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            GGML_ASSERT(st->nb[2] == expert_size && bank->nb[2] == expert_size);

            // one kernel for the layer's tensor: cached experts from the bank, the others from the mapped host experts
            d.r_dst.clear();
            d.r_src.clear();
            for (const auto & r : cp.upload) {
                for (int32_t e = r.first; e < r.first + r.n; ++e) {
                    d.r_dst.push_back((uint8_t *) st->data + (size_t) e*expert_size);
                    d.r_src.push_back(l.ring_host[ip] + (size_t) e*expert_size);
                }
                d.rstats.bytes_host += r.n*expert_size;
            }
            for (int32_t e : cp.cached) {
                d.r_dst.push_back((uint8_t *) st->data + (size_t) e*expert_size);
                d.r_src.push_back((const uint8_t *) bank->data + (size_t) slots[e]*expert_size);
            }
            d.rstats.bytes_cache += cp.cached.size()*expert_size;
            copy_blocks(copy, d.r_dst.data(), d.r_src.data(), (int) d.r_dst.size(), expert_size, ring_blocks);
            d.rstats.launches += (d.r_dst.size() + 127)/128;

            const size_t padding = std::min(expert_size, zeros.size());
            for (int32_t e : cp.pad_next) {
                ggml_backend_tensor_set_async(copy, st, (const uint8_t *) w->data + e*expert_size, e*expert_size, padding);
            }
            if (cp.pad_end) {
                ggml_backend_tensor_set_async(copy, st, d.pinned_zeros, (size_t) n_expert*expert_size, padding);
            }
            d.rstats.copies += cp.pad_next.size() + (cp.pad_end ? 1 : 0);
        }

        for (size_t ip = 0; ip < l.experts.size() && !zc; ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * st   = g.stage[is][ip];
            const ggml_tensor * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            const uint8_t * data = (const uint8_t *) w->data;
            GGML_ASSERT(st->nb[2] == expert_size && bank->nb[2] == expert_size);

            for (const auto & r : cp.upload) {
                ggml_backend_tensor_set_async(copy, st, data + r.first*expert_size, r.first*expert_size, r.n*expert_size);
                d.rstats.bytes_host += r.n*expert_size;
            }
            d.rstats.copies += cp.upload.size();

            if (!cp.cached.empty()) {
                d.r_dst.clear();
                d.r_src.clear();
                for (int32_t e : cp.cached) {
                    d.r_dst.push_back((uint8_t *) st->data + (size_t) e*expert_size);
                    d.r_src.push_back((const uint8_t *) bank->data + (size_t) slots[e]*expert_size);
                }
                if (copy_blocks != nullptr) {
                    copy_blocks(copy, d.r_dst.data(), d.r_src.data(), (int) d.r_dst.size(), expert_size, ring_blocks);
                    d.rstats.launches += (d.r_dst.size() + 127)/128;
                } else {
                    for (size_t k = 0; k < d.r_dst.size(); ++k) {
                        ring_copy_d2d(d, bank, (size_t) slots[cp.cached[k]]*expert_size, st, (size_t) cp.cached[k]*expert_size, expert_size);
                    }
                    d.rstats.copies += d.r_dst.size();
                }
                d.rstats.bytes_cache += cp.cached.size()*expert_size;
            }

            // MMQ can read past the last expert of a run: the start of the next expert, or zeros after the last one
            const size_t padding = std::min(expert_size, zeros.size());
            for (int32_t e : cp.pad_next) {
                ggml_backend_tensor_set_async(copy, st, data + e*expert_size, e*expert_size, padding);
            }
            if (cp.pad_end) {
                ggml_backend_tensor_set_async(copy, st, d.pinned_zeros, (size_t) n_expert*expert_size, padding);
            }
            d.rstats.copies += cp.pad_next.size() + (cp.pad_end ? 1 : 0);
        }

        ggml_backend_event_record(d.ev_ready[is].get(), copy);
        d.last_slot      = is;
        d.slot_layer[is] = -1;

        const int64_t t = ggml_time_us() - t0;
        d.rstats.issue_us    += t;
        d.rstats.issue_us_max = std::max(d.rstats.issue_us_max, t);
    }

    // a device to device copy on the copy stream, without the block copy kernel
    void ring_copy_d2d(device & d, const ggml_tensor * src, size_t src_offs, ggml_tensor * dst, size_t dst_offs, size_t size) {
        ggml_tensor src_view = *src;
        ggml_tensor dst_view = *dst;
        src_view.view_src = nullptr;
        dst_view.view_src = nullptr;
        src_view.ne[0] = dst_view.ne[0] = (int64_t) size;
        src_view.ne[1] = src_view.ne[2] = src_view.ne[3] = dst_view.ne[1] = dst_view.ne[2] = dst_view.ne[3] = 1;
        src_view.type = dst_view.type = GGML_TYPE_I8;
        src_view.nb[0] = dst_view.nb[0] = 1;
        for (int k = 1; k < GGML_MAX_DIMS; ++k) {
            src_view.nb[k] = dst_view.nb[k] = size;
        }
        src_view.data = (uint8_t *) src->data + src_offs;
        dst_view.data = (uint8_t *) dst->data + dst_offs;
        ggml_backend_tensor_copy_async(d.copy.get(), d.copy.get(), &src_view, &dst_view);
    }

    // the graph does not stage the device's layers as planned: the per-layer staging takes over for the rest of the graph
    void ring_stop(device & d) {
        if (d.rs.active()) {
            d.rstats.dropped++;
        }
        d.rs.drop();
        d.ring_fence = true;
        std::fill(d.slot_layer.begin(), d.slot_layer.end(), -1);
    }

    // the staged layer il of the ring: returns false if the ring does not serve it
    bool ring_copy_stage(ggml_backend_t backend, device & d, int32_t il, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        std::lock_guard<std::mutex> lock(*d.mtx);
        if (!d.rs.active()) {
            return false;
        }
        const int64_t i = d.rs.at_layer(il);
        if (i < 0) {
            LLAMA_LOG_WARN("%s: layer %d is not the next layer the MoE ring planned on %s, it stages one layer at a time for this graph\n",
                __func__, il, ggml_backend_name(backend));
            ring_stop(d);
            return false;
        }

        // the matmuls of the layer staged before are queued: its slot can take the next planned layer
        const int32_t rel = d.rs.release();
        if (rel >= 0) {
            ggml_backend_event_record(d.ev_used[rel].get(), backend);
        }

        const llama_moe_ring_sched::entry e = d.rs.plan[i];
        if (!e.issued) {
            if (e.full) {
                ring_fill(d, (size_t) i, nullptr);
                d.rstats.at_layer++;
            } else {
                // the selected experts only: the routing is read back (synchronizes the compute stream)
                const ggml_tensor * sel = find_lookup(graph, dst)->src[1];
                GGML_ASSERT(ggml_is_contiguous(sel));
                const int64_t n_expert = layers[il].stage_map->ne[1];
                d.r_ids.resize(ggml_nelements(sel));
                ggml_backend_tensor_get_async(backend, sel, d.r_ids.data(), 0, ggml_nbytes(sel));
                ggml_backend_synchronize(backend);
                d.r_used.assign(n_expert, false);
                for (int32_t id : d.r_ids) {
                    GGML_ASSERT(id >= 0 && id < n_expert);
                    d.r_used[id] = true;
                }
                ring_fill(d, (size_t) i, &d.r_used);
                d.rstats.partial++;
            }
        }
        ring_issue(d);

        // the compute stream waits only for this layer's slot
        ggml_backend_event_wait(backend, d.ev_ready[e.slot].get());
        if (d.last_slot == e.slot) {
            d.last_slot = -1;
        }
        d.rs.consume((size_t) i);

        // the staging map is the identity
        GGML_ASSERT(ggml_nbytes(src) <= d.pinned_map_size);
        ggml_backend_tensor_set_async(backend, dst, d.pinned_map, 0, ggml_nbytes(src));
        return true;
    }

    bool copy_stage(ggml_backend_t backend, int32_t il, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const layer & l = layers[il];
        device & d = devices[groups[l.ig].id];

        GGML_ASSERT(backend == d.backend);

        if (d.mtx) {
            // the ring did not serve the layer (copy() tried it first)
            std::lock_guard<std::mutex> lock(*d.mtx);
            d.ring_fence = true; // the per-layer staging below changes the slots and the copy stream
        }

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
                // the ring is per device (its own mutex): with submit threads the devices queue their layers in parallel
                device & d = devices[groups[layers[it_stage->second].ig].id];
                if (d.mtx && ring_copy_stage(backend, d, it_stage->second, src, dst, graph)) {
                    return true;
                }
                std::lock_guard<std::mutex> lock(mtx_shared);
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
            if (!stash.empty()) {
                stash.clear();
            }
            return true;
        }
        std::lock_guard<std::mutex> lock(mtx_shared);
        if (d.mtx) {
            // fills and returned slots below: the next ring graph's copies wait for them
            std::lock_guard<std::mutex> lock_dev(*d.mtx);
            ring_stop(d);
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
        if (ids.size() <= (size_t) 8*n_expert_used) {
            observe(il, ids.data(), ids.size());
        }
        const bool planned = ad_on ? plan_adaptive(g, il, ids.data(), ids.size(), n_expert_used, n_hit) :
            g.slru.plan(il, ids.data(), ids.size(), has_cpu ? part->n_used : 1, n_zero, n_cpu, rng, fills, cpu, n_hit);
        if (!planned) {
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
        std::lock_guard<std::mutex> lock(mtx_shared);
        predictions[lookup] = { pred };
    }

    // submit threads: the large-batch path is safe from several threads (one per device); the features that queue work on
    // other streams from the scheduler's thread or keep per-graph state across devices are not
    bool submit_threads_ok(std::string & why) const {
        // the ring is per device (submit_graph); device hits / CPU lane / decode memory act between graphs on the scheduler's
        // thread (submit_settle before the batch, on_alloc after the drain): only the per-layer staging and prefetch are not
        if (n_stage > 0 && ring == 0) { why = "LLAMA_MOE_ASYNC without LLAMA_MOE_RING"; return false; }
        if (n_pref > 0)           { why = "LLAMA_MOE_PREFETCH";          return false; }
        if (cpu_share_q > 0)      { why = "--moe-cache-cpu-share";       return false; }
        if (trace != nullptr)     { why = "LLAMA_MOE_CACHE_TRACE";       return false; }
        return true;
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
        for (const device & d : devices) {
            const auto & r = d.rstats;
            if (r.graphs == 0) {
                continue;
            }
            LLAMA_LOG_INFO("llama_moe_cache: ring %s: %zu graphs, %zu layers streamed ahead, %zu at their layer, %zu partial, %zu graphs dropped; "
                "%zu copies + %zu kernels queued, %.2f GiB uploaded, %.2f GiB from the cache; host %.1f ms queueing (max %.2f ms a layer)\n",
                ggml_backend_name(d.backend), r.graphs, r.layers, r.at_layer, r.partial, r.dropped, r.copies, r.launches,
                r.bytes_host/1024.0/1024.0/1024.0, r.bytes_cache/1024.0/1024.0/1024.0, r.issue_us/1000.0, r.issue_us_max/1000.0);
        }
        if (ad_on) {
            const moe_cache_policy_stats & a = stats_ad;
            LLAMA_LOG_INFO("llama_moe_cache: adaptive policy: %zu fills admitted (%zu into empty slots), %zu misses kept out "
                "(%zu via transient slots, %zu cached anyway), %zu over the cap of %d per token\n",
                a.admitted, a.into_free, a.kept_out, a.transient, a.forced, a.capped, ad_fills_per_token);
        }
        if (dm_on) {
            const dm_stats & m = stats_dm;
            LLAMA_LOG_INFO("llama_moe_cache: decode memory: %zu large-batch takes (%.1f ms each, %.2f MiB released per take), %zu gives "
                "(%.1f ms each), %zu experts moved, %zu evicted, %zu refilled (%.2f MiB) in %zu graph-devices\n",
                m.takes, m.takes ? m.take_us/1000.0/m.takes : 0.0, m.takes ? m.released/1024.0/1024.0/m.takes : 0.0,
                m.gives, m.gives ? m.give_us/1000.0/m.gives : 0.0, m.moved, m.evicted, m.refilled, m.refill_bytes/1024.0/1024.0, m.refill_graphs);
            auto cost = [](const char * what, const dm_cost & c, size_t n) {
                if (n > 0) {
                    LLAMA_LOG_INFO("llama_moe_cache: decode memory: %s ms each: sync %.1f, evict %.1f, tails %.1f, compute %.1f, staging %.1f, rest %.1f\n",
                        what, c.sync/1000.0/n, c.evict/1000.0/n, c.tail/1000.0/n, c.comp/1000.0/n, c.stage/1000.0/n, c.rest/1000.0/n);
                }
            };
            cost("take", m.take_cost, m.takes);
            cost("give", m.give_cost, m.gives);
            if (m.tails_out > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: decode memory: %zu times a group tail stayed out after a give (no memory)\n", m.tails_out);
            }
            for (size_t id = 0; id < dm_devs.size(); ++id) {
                const dm_device & dm = dm_devs[id];
                if (dm.lend > 0) {
                    LLAMA_LOG_INFO("llama_moe_cache: decode memory: %s compute used: small graphs up to %.2f MiB (kept %.2f), large up to %.2f MiB (measured %.2f)\n",
                        ggml_backend_name(devices[id].backend), dm.used_small_max/1024.0/1024.0, dm.small/1024.0/1024.0,
                        dm.used_large_max/1024.0/1024.0, dm.large/1024.0/1024.0);
                }
            }
        }
        const dh_stats & ds = stats_dh;
        if (ds.hits + ds.misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: device hits: %zu graphs, hits = %zu, misses = %zu read from host memory (%.2f MiB), hit rate = %.2f%%, "
                "fills = %zu (%.2f MiB), %zu dropped\n", ds.graphs, ds.hits, ds.misses, ds.bytes_miss/1024.0/1024.0,
                100.0*ds.hits/(ds.hits + ds.misses), ds.fills, ds.bytes_fill/1024.0/1024.0, ds.dropped);
        }
        if (lane) {
            const llama_moe_lane::stats ls = lane->get_stats();
            if (ls.requests > 0 || ls.timeouts > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: CPU lane: %" PRIu64 " requests, %" PRIu64 " experts (%.2f per request), %" PRIu64 " rows, "
                    "%.3f ms per request, %" PRIu64 " timeouts\n", ls.requests, ls.experts, (double) ls.experts/std::max<uint64_t>(1, ls.requests),
                    ls.rows, ls.busy_ms/std::max<uint64_t>(1, ls.requests), ls.timeouts);
            }
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share,
        const std::string & profile, const llama_moe_cache_compute & compute) :
    pimpl(new impl(model, backends, bufts, size, cpu_share, profile, compute)) {
}

bool llama_moe_cache::wants_compute_sizes() {
    const char * dm    = getenv("LLAMA_MOE_DECODE_MEM");
    const char * split = getenv("LLAMA_MOE_CACHE_SPLIT");
    return (dm != nullptr && atoi(dm) != 0) || (split != nullptr && (strcmp(split, "miss") == 0 || strcmp(split, "stream") == 0));
}

int32_t llama_moe_cache::small_batch_tokens() {
    const char * env = getenv("LLAMA_MOE_DECODE_MEM_TOKENS");
    return env != nullptr ? std::max(1, atoi(env)) : 32;
}

ggml_backend_buffer_type_t llama_moe_cache::compute_buft(ggml_backend_t backend) const {
    return pimpl->compute_buft(backend);
}

float llama_moe_cache::compact_ratio() const {
    return pimpl->dm_on ? pimpl->dm_compact : 0.0f;
}

void llama_moe_cache::reserved() {
    pimpl->graph_stages = false;
}

void llama_moe_cache::on_alloc(ggml_backend_sched_t sched) {
    pimpl->on_alloc(sched);
}

llama_moe_cache::~llama_moe_cache() = default;

bool llama_moe_cache::submit_threads_ok(std::string & why) const {
    return pimpl->submit_threads_ok(why);
}

void llama_moe_cache::submit_settle() {
    pimpl->submit_settle();
}

void llama_moe_cache::ring_plan_copy(ggml_cgraph * graph, int copy) {
    pimpl->ring_plan_copy(graph, copy);
}

void llama_moe_cache::submit_graph(ggml_backend_t backend, int copy, bool begin) {
    pimpl->submit_graph(backend, copy, begin);
}

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

void * llama_moe_cache::cpu_lane(int32_t il, int64_t n_tokens, int64_t n_expert_used, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * down, const ggml_tensor * gate_up, float clamp) {
    return pimpl->cpu_lane(il, n_tokens, n_expert_used, up, gate, down, gate_up, clamp);
}

float llama_moe_cache::cpu_lane_frac() const {
    return pimpl->lane_frac;
}

int32_t llama_moe_cache::cpu_lane_max_tokens() const {
    return (int32_t) pimpl->dh_max;
}

int32_t llama_moe_cache::cpu_lane_timeout_ms() const {
    return pimpl->lane_timeout_ms;
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

const void * llama_moe_cache::cached_expert(ggml_backend_t backend, const ggml_tensor * w, int64_t e, size_t size) {
    return pimpl->cached_expert(backend, w, e, size);
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
    for (const auto & b : pimpl->lane_bufs) {
        res[ggml_backend_buffer_get_type(b.get())] += ggml_backend_buffer_get_size(b.get());
    }
    return res;
}
