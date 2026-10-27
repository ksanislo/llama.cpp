#pragma once

#include "llama.h"
#include "llama-graph.h"

#include <map>
#include <memory>
#include <functional>
#include <vector>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;
class llama_memory_recurrent;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    llama_memory_t mem_other;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;

    // prompt pipe with stable batch shapes (LLAMA_PROMPT_PIPE_CAUSAL), after apply(): true gives the current ubatch the
    // shapes of the batch's end (the layout the batch is planned in), false its own causal ones (n_kv / pools so far)
    // returns false if the shapes do not depend on it
    virtual bool set_shapes_plan(bool /*plan*/) { return false; }
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    // embd_all: keep all the tokens of a sequence in one ubatch (all tokens are output, or the attention is non-causal)
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // elastic YaRN (llama_cparams::rope_elastic): every cached cell remembers the rope level its key was
    // rotated with, and a sequence can move its cached keys to another level
    //

    // prompt pipe: the ubatches of the next batches keep one graph shape (the sizes of the batch's last ubatch), so the
    // compute buffers planned for the first fit the others and a pipelined batch never reallocates them
    virtual void set_batch_shapes_stable(bool /*stable*/) {}

    // track the rope levels; false if this memory cannot re-rotate its cached keys
    virtual bool rope_level_enable() { return false; }

    // the rope level of the cells of seq_id, -1 if it has none
    virtual int32_t seq_rope_level(llama_seq_id /*seq_id*/) const { return -1; }

    // seq_id plus every sequence sharing a cell with it, directly or through others: they can only move together
    virtual std::vector<llama_seq_id> seq_rope_group(llama_seq_id seq_id) const { return { seq_id }; }

    // move the cached keys of seq_id to level, done by the next init_update(); pass a whole seq_rope_group()
    virtual void seq_rope_level_set(llama_seq_id /*seq_id*/, int32_t /*level*/) {}

    // the recurrent memory whose state is ALL that state_write(seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) writes
    // (the other parts write nothing then), or nullptr: the partial state can then be checkpointed mid-batch
    virtual llama_memory_recurrent * partial_state_recurrent() const { return nullptr; }

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
