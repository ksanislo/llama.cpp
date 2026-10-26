#include "server-session.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// same constants as the /slots save path (server-context.cpp reads them back)
static constexpr uint32_t SLOT_CKPT_MAGIC    = 0x504b4353; // "SCKP"
static constexpr uint32_t SLOT_CKPT_VERSION  = 1;
static constexpr uint32_t SLOT_DRAFT_MAGIC   = 0x54464453; // "SDFT"
static constexpr uint32_t SLOT_DRAFT_VERSION = 1;

// trailer at the very end of a session file: [identity][u32 n_ckpt][i64 x n_ckpt][u32 trailer size][u32 magic]
static constexpr uint32_t SESSION_MAGIC = 0x4e535353; // "SSSN"

// llama_state_seq_get_data's header (llama-context.cpp io_magic)
static constexpr uint32_t SEQ_STATE_IO_MAGIC = 0xaf143cd8;

static const char * SESSION_EXT = ".session";

static void os_write(std::ostream & os, const void * src, size_t size, size_t & n_written) {
    os.write((const char *) src, size);
    n_written += size;
}

static void os_write_buf(std::ostream & os, const std::vector<uint8_t> & buf, size_t & n_written) {
    const uint64_t n = buf.size();
    os_write(os, &n, sizeof(n), n_written);
    if (n > 0) {
        os_write(os, buf.data(), n, n_written);
    }
}

void server_slot_write_checkpoints(std::ostream & os, const std::list<common_prompt_checkpoint> & checkpoints, size_t & n_written) {
    if (checkpoints.empty()) {
        return;
    }
    const uint32_t magic   = SLOT_CKPT_MAGIC;
    const uint32_t version = SLOT_CKPT_VERSION;
    const uint32_t count   = (uint32_t) checkpoints.size();
    os_write(os, &magic,   sizeof(magic),   n_written);
    os_write(os, &version, sizeof(version), n_written);
    os_write(os, &count,   sizeof(count),   n_written);
    for (const auto & cur : checkpoints) {
        os_write(os, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
        os_write(os, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
        os_write(os, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
        os_write_buf(os, cur.data_tgt,  n_written);
        os_write_buf(os, cur.data_dft,  n_written);
        os_write_buf(os, cur.data_spec, n_written);
    }
}

void server_slot_write_draft(std::ostream & os, const std::vector<uint8_t> & data_dft, const std::vector<uint8_t> & data_spec, size_t & n_written) {
    if (data_dft.empty() && data_spec.empty()) {
        return;
    }
    const uint32_t magic   = SLOT_DRAFT_MAGIC;
    const uint32_t version = SLOT_DRAFT_VERSION;
    os_write(os, &magic,   sizeof(magic),   n_written);
    os_write(os, &version, sizeof(version), n_written);
    os_write_buf(os, data_dft,  n_written);
    os_write_buf(os, data_spec, n_written);
}

size_t server_session_job::size() const {
    size_t res = packed.size() + state.size() + data_dft.size() + data_spec.size();
    for (const auto & c : checkpoints) {
        res += c.size();
    }
    return res;
}

static int64_t file_mtime_s(const fs::path & p) {
    std::error_code ec;
    const auto t = fs::last_write_time(p, ec);
    if (ec) {
        return 0;
    }
    const auto sys = std::chrono::time_point_cast<std::chrono::seconds>(
        t - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    return sys.time_since_epoch().count();
}

static int64_t now_s() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// FNV-1a over the packed tokens: the file name of a conversation state
static std::string session_key(const std::vector<char> & packed) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (char c : packed) {
        h ^= (uint8_t) c;
        h *= 0x100000001b3ull;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%016" PRIx64, h);
    return buf;
}

server_session_dir::server_session_dir(const server_session_params & params, const std::string & identity, bool has_mtmd)
    : params(params), identity(identity), has_mtmd(has_mtmd) {
    std::error_code ec;
    fs::create_directories(fs::u8path(params.dir), ec);
    scan();
    worker = std::thread([this]() { run(); });
}

server_session_dir::~server_session_dir() {
    flush();
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

bool server_session_dir::read_header(const std::string & path, server_session_entry & entry, std::string & err) const {
    std::ifstream ifs(fs::u8path(path), std::ios::binary | std::ios::ate);
    if (!ifs) {
        err = "cannot open";
        return false;
    }
    const uint64_t file_size = (uint64_t) ifs.tellg();
    if (file_size < 20) {
        err = "too short";
        return false;
    }

    // trailer
    uint32_t tr[2] = {0, 0};
    ifs.seekg(file_size - 8);
    if (!ifs.read((char *) tr, 8) || tr[1] != SESSION_MAGIC || tr[0] < 4 || tr[0] + 8 > file_size) {
        err = "no session trailer";
        return false;
    }
    std::string trailer(tr[0], '\0');
    ifs.seekg(file_size - 8 - tr[0]);
    if (!ifs.read(trailer.data(), tr[0])) {
        err = "truncated trailer";
        return false;
    }
    uint32_t n_ckpt = 0;
    if (trailer.size() < identity.size() + 4 || trailer.compare(0, identity.size(), identity) != 0) {
        err = "foreign";
        return false;
    }
    memcpy(&n_ckpt, trailer.data() + identity.size(), 4);
    if (identity.size() + 4 + (size_t) n_ckpt*8 != trailer.size()) {
        err = "foreign";
        return false;
    }
    entry.ckpt_n_tokens.resize(n_ckpt);
    if (n_ckpt > 0) {
        memcpy(entry.ckpt_n_tokens.data(), trailer.data() + identity.size() + 4, (size_t) n_ckpt*8);
    }

    // header: magic, version, token count, packed tokens
    uint32_t hdr[3] = {0, 0, 0};
    ifs.seekg(0);
    if (!ifs.read((char *) hdr, sizeof(hdr)) || hdr[0] != LLAMA_STATE_SEQ_MAGIC || hdr[1] != LLAMA_STATE_SEQ_VERSION) {
        err = "not a sequence state file";
        return false;
    }
    if ((uint64_t) hdr[2] * sizeof(llama_token) + 12 > file_size) {
        err = "bad token count";
        return false;
    }
    llama_tokens packed(hdr[2]);
    if (hdr[2] > 0 && !ifs.read((char *) packed.data(), (size_t) hdr[2] * sizeof(llama_token))) {
        err = "truncated tokens";
        return false;
    }
    try {
        entry.tokens = server_tokens::deserialize(packed, has_mtmd);
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }

    entry.path  = path;
    entry.bytes = file_size;
    entry.mtime = file_mtime_s(fs::u8path(path));
    return true;
}

void server_session_dir::scan() {
    std::error_code ec;
    int n_foreign = 0;
    int n_bad     = 0;
    for (const auto & de : fs::directory_iterator(fs::u8path(params.dir), ec)) {
        if (!de.is_regular_file(ec)) {
            continue;
        }
        const std::string path = de.path().u8string();
        if (de.path().extension() == ".tmp") {
            fs::remove(de.path(), ec); // a write that did not finish
            continue;
        }
        if (de.path().extension() != SESSION_EXT) {
            continue;
        }
        server_session_entry e;
        std::string err;
        if (!read_header(path, e, err)) {
            if (err == "foreign") {
                n_foreign++;
            } else {
                n_bad++;
                SRV_WRN("sessions: skipping '%s': %s\n", path.c_str(), err.c_str());
            }
            continue;
        }
        entries.push_back(std::move(e));
    }
    entries.sort([](const server_session_entry & a, const server_session_entry & b) { return a.mtime < b.mtime; });

    std::lock_guard<std::mutex> lock(mtx);
    enforce_limits_locked("");

    size_t bytes = 0;
    for (const auto & e : entries) {
        bytes += e.bytes;
    }
    SRV_INF("sessions: %zu saved conversations in '%s' (%.2f GiB)%s\n", entries.size(), params.dir.c_str(), bytes / 1073741824.0,
            n_foreign ? string_format(", %d of another model or build left alone", n_foreign).c_str() : "");
    (void) n_bad;
}

void server_session_dir::remove_entry_locked(std::list<server_session_entry>::iterator it, const char * why) {
    std::error_code ec;
    fs::remove(fs::u8path(it->path), ec);
    SRV_INF("sessions: removed %zu-token conversation (%.2f GiB): %s\n", it->tokens.size(), it->bytes / 1073741824.0, why);
    entries.erase(it);
}

void server_session_dir::enforce_limits_locked(const std::string & keep) {
    // entries are in LRU order, oldest first
    const int64_t t_min = params.max_days > 0 ? now_s() - (int64_t) params.max_days * 86400 : INT64_MIN;
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->mtime < t_min && it->path != keep) {
            auto cur = it++;
            remove_entry_locked(cur, string_format("unused for %d days", params.max_days).c_str());
        } else {
            ++it;
        }
    }
    auto total = [&]() { size_t r = 0; for (const auto & e : entries) { r += e.bytes; } return r; };
    while (!entries.empty() &&
           ((params.max_bytes > 0 && total() > params.max_bytes) || (params.max_files > 0 && entries.size() > (size_t) params.max_files))) {
        auto it = entries.begin();
        if (it->path == keep && entries.size() > 1) {
            ++it;
        }
        remove_entry_locked(it, "least recently used");
    }
}

bool server_session_dir::write_job(server_session_job & job, std::string & path, size_t & bytes) {
    path = (fs::u8path(params.dir) / fs::u8path(session_key(job.packed) + SESSION_EXT)).u8string();
    const std::string tmp = path + ".tmp";

    const int64_t t0 = ggml_time_us();
    size_t n = 0;
    {
        std::ofstream ofs(fs::u8path(tmp), std::ios::binary | std::ios::trunc);
        if (!ofs) {
            SRV_WRN("sessions: cannot create '%s'\n", tmp.c_str());
            return false;
        }
        const uint32_t hdr[3] = { LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION, (uint32_t) (job.packed.size() / sizeof(llama_token)) };
        os_write(ofs, hdr, sizeof(hdr), n);
        os_write(ofs, job.packed.data(), job.packed.size(), n);
        // the in-memory state starts with an io magic + seq id that the file format does not have
        size_t skip = 0;
        if (job.state.size() >= 8 && *(const uint32_t *) job.state.data() == SEQ_STATE_IO_MAGIC) {
            skip = 8;
        }
        os_write(ofs, job.state.data() + skip, job.state.size() - skip, n);
        server_slot_write_checkpoints(ofs, job.checkpoints, n);
        server_slot_write_draft(ofs, job.data_dft, job.data_spec, n);

        std::string trailer = identity;
        const uint32_t n_ckpt = (uint32_t) job.checkpoints.size();
        trailer.append((const char *) &n_ckpt, 4);
        for (const auto & c : job.checkpoints) {
            const int64_t nt = c.n_tokens;
            trailer.append((const char *) &nt, 8);
        }
        const uint32_t tr[2] = { (uint32_t) trailer.size(), SESSION_MAGIC };
        os_write(ofs, trailer.data(), trailer.size(), n);
        os_write(ofs, tr, sizeof(tr), n);
        ofs.flush();
        if (!ofs) {
            std::error_code ec;
            fs::remove(fs::u8path(tmp), ec);
            SRV_WRN("sessions: writing '%s' failed (disk full?)\n", tmp.c_str());
            return false;
        }
    }
    std::error_code ec;
    fs::rename(fs::u8path(tmp), fs::u8path(path), ec);
    if (ec) {
        fs::remove(fs::u8path(tmp), ec);
        return false;
    }
    bytes = n;
    const double dt = (ggml_time_us() - t0) / 1e6;
    SRV_INF("sessions: saved a %zu-token conversation in the background (%.2f GiB, %.1f s)\n", job.tokens.size(), n / 1073741824.0, dt);
    return true;
}

void server_session_dir::add_entry_locked(server_session_job && job, const std::string & path, size_t bytes) {
    // a newer state of the same conversation replaces the older one
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->path != path && it->tokens.size() <= job.tokens.size() &&
            it->tokens.get_common_prefix(job.tokens) == it->tokens.size()) {
            auto cur = it++;
            remove_entry_locked(cur, "replaced by its continuation");
        } else if (it->path == path) {
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
    server_session_entry e;
    e.path  = path;
    e.tokens = std::move(job.tokens);
    for (const auto & c : job.checkpoints) {
        e.ckpt_n_tokens.push_back(c.n_tokens);
    }
    e.bytes = bytes;
    e.mtime = now_s();
    entries.push_back(std::move(e));
    enforce_limits_locked(path);
}

void server_session_dir::run() {
    std::unique_lock<std::mutex> lock(mtx);
    while (true) {
        if (pending.empty()) {
            cv_done.notify_all();
            if (stop) {
                return;
            }
            cv.wait(lock);
            continue;
        }
        auto it = std::min_element(pending.begin(), pending.end(),
                [](const server_session_job & a, const server_session_job & b) { return a.t_due_us < b.t_due_us; });
        const int64_t t_now = ggml_time_us();
        if (!flushing && !stop && it->t_due_us > t_now && n_pending_bytes <= params.max_pending) {
            cv.wait_for(lock, std::chrono::microseconds(it->t_due_us - t_now));
            continue;
        }
        server_session_job job = std::move(*it);
        pending.erase(it);
        n_pending_bytes -= std::min(n_pending_bytes, job.size());
        writing = true;
        lock.unlock();

        std::string path;
        size_t bytes = 0;
        const bool ok = write_job(job, path, bytes);

        lock.lock();
        writing = false;
        if (ok) {
            add_entry_locked(std::move(job), path, bytes);
        }
        cv_done.notify_all();
    }
}

void server_session_dir::push(server_session_job && job) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        // an older pending snapshot of the same conversation is superseded
        for (auto it = pending.begin(); it != pending.end();) {
            if (it->tokens.size() <= job.tokens.size() && it->tokens.get_common_prefix(job.tokens) == it->tokens.size()) {
                n_pending_bytes -= std::min(n_pending_bytes, it->size());
                it = pending.erase(it);
            } else {
                ++it;
            }
        }
        n_pending_bytes += job.size();
        pending.push_back(std::move(job));
    }
    cv.notify_all();
}

bool server_session_dir::has(const server_tokens & tokens) {
    std::lock_guard<std::mutex> lock(mtx);
    for (const auto & e : entries) {
        if (e.tokens.size() == tokens.size() && e.tokens.get_common_prefix(tokens) == tokens.size()) {
            return true;
        }
    }
    for (const auto & j : pending) {
        if (j.tokens.size() == tokens.size() && j.tokens.get_common_prefix(tokens) == tokens.size()) {
            return true;
        }
    }
    return false;
}

void server_session_dir::cancel(const server_tokens & prompt, const server_tokens & held) {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto it = pending.begin(); it != pending.end();) {
        const size_t n_use = it->tokens.get_common_prefix(prompt);
        if (n_use >= it->tokens.size() / 2 && it->tokens.get_common_prefix(held) >= n_use) {
            SRV_INF("sessions: %zu-token conversation taken back before it was written - not written\n", it->tokens.size());
            n_pending_bytes -= std::min(n_pending_bytes, it->size());
            it = pending.erase(it);
        } else {
            ++it;
        }
    }
}

void server_session_dir::flush() {
    std::unique_lock<std::mutex> lock(mtx);
    flushing = true;
    cv.notify_all();
    cv_done.wait(lock, [&]() { return pending.empty() && !writing; });
    flushing = false;
}

static size_t usable_prefix(const server_tokens & tokens, const std::vector<int64_t> & ckpts, size_t lcp, bool rm_full) {
    if (lcp >= tokens.size() || rm_full) {
        return lcp;
    }
    size_t best = 0;
    for (int64_t n : ckpts) {
        if (n > 0 && (size_t) n <= lcp) {
            best = std::max(best, (size_t) n);
        }
    }
    return best;
}

bool server_session_dir::find(const server_tokens & prompt, bool rm_full, size_t min_usable, std::string & path, size_t & n_usable, size_t & n_tokens) {
    std::unique_lock<std::mutex> lock(mtx);

    size_t best = 0;
    auto it_best = entries.end();
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        const size_t u = usable_prefix(it->tokens, it->ckpt_n_tokens, it->tokens.get_common_prefix(prompt), rm_full);
        if (u > best) {
            best = u;
            it_best = it;
        }
    }

    auto jt_best = pending.end();
    for (auto jt = pending.begin(); jt != pending.end(); ++jt) {
        std::vector<int64_t> ck;
        for (const auto & c : jt->checkpoints) {
            ck.push_back(c.n_tokens);
        }
        const size_t u = usable_prefix(jt->tokens, ck, jt->tokens.get_common_prefix(prompt), rm_full);
        if (u > best) {
            best = u;
            jt_best = jt;
        }
    }

    if (best < min_usable || best == 0) {
        return false;
    }

    if (jt_best != pending.end()) {
        // the conversation is only in memory so far: write it now, on this thread
        server_session_job job = std::move(*jt_best);
        pending.erase(jt_best);
        n_pending_bytes -= std::min(n_pending_bytes, job.size());
        lock.unlock();
        std::string p;
        size_t bytes = 0;
        if (!write_job(job, p, bytes)) {
            return false;
        }
        lock.lock();
        n_tokens = job.tokens.size();
        add_entry_locked(std::move(job), p, bytes);
        path     = p;
        n_usable = best;
        return true;
    }

    path     = it_best->path;
    n_usable = best;
    n_tokens = it_best->tokens.size();
    return true;
}

void server_session_dir::touch(const std::string & path) {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->path == path) {
            std::error_code ec;
            fs::last_write_time(fs::u8path(path), fs::file_time_type::clock::now(), ec);
            it->mtime = now_s();
            entries.splice(entries.end(), entries, it);
            return;
        }
    }
}

void server_session_dir::remove(const std::string & path) {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->path == path) {
            remove_entry_locked(it, "could not be restored");
            return;
        }
    }
}

json server_session_dir::status() {
    std::lock_guard<std::mutex> lock(mtx);
    json convs = json::array();
    size_t bytes = 0;
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        bytes += it->bytes;
        convs.push_back({
            {"tokens",    it->tokens.size()},
            {"bytes",     it->bytes},
            {"last_used", it->mtime},
            {"state",     "on disk"},
        });
    }
    for (const auto & j : pending) {
        convs.push_back({
            {"tokens", j.tokens.size()},
            {"bytes",  j.size()},
            {"state",  "waiting to be written"},
        });
    }
    return json {
        {"dir",          params.dir},
        {"files",        entries.size()},
        {"bytes",        bytes},
        {"max_bytes",    params.max_bytes},
        {"max_files",    params.max_files},
        {"max_age_days", params.max_days},
        {"pending",      pending.size()},
        {"conversations", convs},
    };
}
