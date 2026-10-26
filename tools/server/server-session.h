#pragma once

// --session-dir: every conversation persisted to disk, written in the background, read back on demand
//
// A file is a slot save file (/slots?action=save format: llama sequence state, then the context checkpoint
// and draft appendices) followed by a trailer with the server identity (model, cache types, build) and the
// checkpoint positions, so the index can be built from the headers and trailers alone.

#include "common.h"
#include "server-common.h"

#include <condition_variable>
#include <cstdint>
#include <list>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

// appendices shared with the /slots save path
void server_slot_write_checkpoints(std::ostream & os, const std::list<common_prompt_checkpoint> & checkpoints, size_t & n_written);
void server_slot_write_draft(std::ostream & os, const std::vector<uint8_t> & data_dft, const std::vector<uint8_t> & data_spec, size_t & n_written);

// a conversation snapshot waiting to be written
struct server_session_job {
    server_tokens tokens;          // for the index
    std::vector<char> packed;      // server_tokens::serialize()
    std::vector<uint8_t> state;    // llama_state_seq_get_data_ext of the target context
    std::list<common_prompt_checkpoint> checkpoints;
    std::vector<uint8_t> data_dft;
    std::vector<uint8_t> data_spec;

    int64_t t_due_us = 0;

    size_t size() const;
};

struct server_session_entry {
    std::string path;
    server_tokens tokens;
    std::vector<int64_t> ckpt_n_tokens;
    size_t  bytes = 0;
    int64_t mtime = 0; // seconds
};

struct server_session_params {
    std::string dir;
    size_t  max_bytes   = 64ull << 30;
    int32_t max_files   = 64;
    int32_t max_days    = 7;
    int32_t delay_s     = 30;
    int32_t min_tokens  = 512;
    size_t  max_pending = 16ull << 30; // pending snapshots beyond this are written at once
};

class server_session_dir {
public:
    server_session_dir(const server_session_params & params, const std::string & identity, bool has_mtmd);
    ~server_session_dir();

    const server_session_params & get_params() const { return params; }

    // usable prefix of the best file (or pending snapshot) for a prompt
    // rm_full: the context can truncate any state, otherwise only at a saved checkpoint or the file's end
    // a pending snapshot that wins is written at once, so the result is always a file
    bool find(const server_tokens & prompt, bool rm_full, size_t min_usable, std::string & path, size_t & n_usable, size_t & n_tokens);

    // drop pending snapshots whose part the prompt uses is held by a slot again (the conversation is back)
    void cancel(const server_tokens & prompt, const server_tokens & held);

    void push(server_session_job && job);

    // a file or pending snapshot holds exactly these tokens
    bool has(const server_tokens & tokens);

    // write everything pending now and wait for it
    void flush();

    // a resumed file goes to the back of the LRU order
    void touch(const std::string & path);

    // a file that could not be restored
    void remove(const std::string & path);

    json status();

private:
    server_session_params params;
    std::string identity;
    bool has_mtmd;

    std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable cv_done;
    std::list<server_session_job> pending;
    std::list<server_session_entry> entries;
    size_t n_pending_bytes = 0;
    bool   writing = false;
    bool   stop    = false;
    bool   flushing = false;
    std::thread worker;

    void scan();
    void run();
    bool write_job(server_session_job & job, std::string & path, size_t & bytes);
    void add_entry_locked(server_session_job && job, const std::string & path, size_t bytes);
    void enforce_limits_locked(const std::string & keep);
    void remove_entry_locked(std::list<server_session_entry>::iterator it, const char * why);
    bool read_header(const std::string & path, server_session_entry & entry, std::string & err) const;
};
