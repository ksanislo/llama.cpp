// NUMA interleave of host memory, see ggml_numa_set_interleave() in ggml.h

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "ggml-impl.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

// from <numaif.h>, without depending on libnuma
#define GGML_MPOL_DEFAULT     0
#define GGML_MPOL_INTERLEAVE  3
#define GGML_MPOL_MF_MOVE     (1 << 1)
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ    22
#endif

#define GGML_NUMA_MASK_LONGS  16 // up to 1024 nodes
#define GGML_NUMA_MAXNODE     (GGML_NUMA_MASK_LONGS*64 + 1) // the kernel reads maxnode - 1 bits

static unsigned long g_numa_mask[GGML_NUMA_MASK_LONGS];
static int           g_numa_n_nodes = -1;
#endif

static int g_numa_interleave = -1; // -1: not decided yet, then from GGML_NUMA_INTERLEAVE

#if defined(__linux__)
// nodes with memory ("0-1", "0,2-3"); memoryless nodes cannot hold an interleaved page
static int ggml_numa_parse_nodes(const char * path) {
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }
    char buf[1024] = { 0 };
    const bool ok = fgets(buf, sizeof(buf), f) != NULL;
    fclose(f);
    if (!ok) {
        return 0;
    }
    memset(g_numa_mask, 0, sizeof(g_numa_mask));
    int n = 0;
    for (char * q = buf; *q >= '0' && *q <= '9';) {
        char * e = NULL;
        const long a = strtol(q, &e, 10);
        long b = a;
        if (*e == '-') {
            b = strtol(e + 1, &e, 10);
        }
        for (long k = a; k <= b && k < GGML_NUMA_MASK_LONGS*64; ++k, ++n) {
            g_numa_mask[k / 64] |= 1ul << (k % 64);
        }
        q = *e == ',' ? e + 1 : e;
    }
    return n;
}

static int ggml_numa_n_nodes(void) {
    if (g_numa_n_nodes < 0) {
        int n = ggml_numa_parse_nodes("/sys/devices/system/node/has_memory");
        if (n == 0) {
            n = ggml_numa_parse_nodes("/sys/devices/system/node/online");
        }
        g_numa_n_nodes = n;
    }
    return g_numa_n_nodes;
}

static size_t ggml_numa_page_size(void) {
    const long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? (size_t) p : 4096;
}

static long ggml_numa_mbind(void * addr, size_t size, unsigned flags) {
    return syscall(SYS_mbind, addr, size, GGML_MPOL_INTERLEAVE, g_numa_mask, GGML_NUMA_MAXNODE, flags);
}
#endif

void ggml_numa_set_interleave(bool enable) {
    g_numa_interleave = enable ? 1 : 0;
    // log now: the first use may come while the caller silences the log (llama's parameter fitting)
    (void) ggml_numa_interleave_enabled();
}

int ggml_numa_free_bytes(int64_t * free_per_node, int max_nodes) {
#if defined(__linux__)
    int n = 0;
    for (; n < max_nodes; ++n) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", n);
        FILE * f = fopen(path, "r");
        if (f == NULL) {
            break;
        }
        char line[256];
        long long kb = -1;
        while (fgets(line, sizeof(line), f)) {
            const char * p = strstr(line, "MemFree:");
            if (p != NULL) {
                kb = strtoll(p + 8, NULL, 10);
                break;
            }
        }
        fclose(f);
        free_per_node[n] = kb < 0 ? -1 : (int64_t) kb * 1024;
    }
    return n;
#else
    GGML_UNUSED(free_per_node);
    GGML_UNUSED(max_nodes);
    return 0;
#endif
}

bool ggml_numa_interleave_enabled(void) {
    if (g_numa_interleave < 0) {
        const char * env = getenv("GGML_NUMA_INTERLEAVE");
        g_numa_interleave = env != NULL && env[0] != '\0' && strcmp(env, "0") != 0;
    }
    if (!g_numa_interleave) {
        return false;
    }
#if defined(__linux__)
    static bool logged = false;
    const int n = ggml_numa_n_nodes();
    if (!logged) {
        logged = true;
        if (n > 1) {
            GGML_LOG_INFO("%s: host weight buffers are interleaved over %d NUMA nodes\n", __func__, n);
        } else {
            GGML_LOG_INFO("%s: only one NUMA node, interleave has no effect\n", __func__);
        }
    }
    return n > 1;
#else
    return false;
#endif
}

bool ggml_numa_interleave(void * addr, size_t size) {
#if defined(__linux__)
    if (addr == NULL || size == 0 || !ggml_numa_interleave_enabled()) {
        return false;
    }
    const size_t page = ggml_numa_page_size();
    const uintptr_t beg = (uintptr_t) addr & ~(uintptr_t) (page - 1);
    const uintptr_t end = ((uintptr_t) addr + size + page - 1) & ~(uintptr_t) (page - 1);
    if (ggml_numa_mbind((void *) beg, end - beg, GGML_MPOL_MF_MOVE) != 0) {
        GGML_LOG_WARN("%s: mbind of %.2f MiB failed: %s\n", __func__, size/1024.0/1024.0, strerror(errno));
        return false;
    }
    return true;
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(size);
    return false;
#endif
}

void * ggml_numa_interleave_alloc(size_t size) {
#if defined(__linux__)
    if (size == 0 || !ggml_numa_interleave_enabled()) {
        return NULL;
    }
    const size_t page = ggml_numa_page_size();
    const size_t len  = (size + page - 1) & ~(page - 1);
    void * ptr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        return NULL;
    }
    // no page has been touched yet, so nothing needs to move
    if (ggml_numa_mbind(ptr, len, 0) != 0) {
        GGML_LOG_WARN("%s: mbind of %.2f MiB failed: %s\n", __func__, size/1024.0/1024.0, strerror(errno));
        munmap(ptr, len);
        return NULL;
    }
    return ptr;
#else
    GGML_UNUSED(size);
    return NULL;
#endif
}

void ggml_numa_interleave_free(void * ptr, size_t size) {
#if defined(__linux__)
    if (ptr != NULL) {
        const size_t page = ggml_numa_page_size();
        munmap(ptr, (size + page - 1) & ~(page - 1));
    }
#else
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
#endif
}

#if defined(__linux__)
static void ggml_numa_populate(void * addr, size_t size, size_t page) {
    if (madvise(addr, size, MADV_POPULATE_READ) == 0) {
        return;
    }
    // before Linux 5.14, or interrupted: read a byte of every page
    volatile const char * p = (volatile const char *) addr;
    for (size_t off = 0; off < size; off += page) {
        (void) p[off];
    }
}

// share of the busiest node among a sample of the range's resident pages (0: too few resident pages to tell)
static double ggml_numa_busiest_share(void * addr, size_t size, size_t page) {
    enum { n_batch = 256, n_batches = 16 };
    const size_t n_pages = size / page;
    if (n_pages == 0) {
        return 0.0;
    }
    long counts[64] = { 0 };
    long total = 0;
    uint64_t x = 0x9e3779b97f4a7c15ull ^ (uint64_t) (uintptr_t) addr;
    for (int b = 0; b < n_batches; ++b) {
        void * pages [n_batch];
        int    status[n_batch];
        for (int i = 0; i < n_batch; ++i) {
            // random pages: a fixed stride can alias with the round-robin of large folios
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            pages[i] = (char *) addr + (x % n_pages) * page;
        }
        if (syscall(SYS_move_pages, 0, (unsigned long) n_batch, pages, NULL, status, 0) != 0) {
            return 0.0;
        }
        for (int i = 0; i < n_batch; ++i) {
            if (status[i] >= 0 && status[i] < 64) {
                counts[status[i]]++;
                total++;
            }
        }
    }
    if (total < 64) {
        return 0.0;
    }
    long busiest = 0;
    for (int i = 0; i < 64; ++i) {
        busiest = counts[i] > busiest ? counts[i] : busiest;
    }
    return (double) busiest / total;
}
#endif

#if defined(__linux__)
static _Thread_local int           g_numa_old_mode = -1; // saved by ggml_numa_thread_interleave_begin, -1: nothing saved
static _Thread_local unsigned long g_numa_old_mask[GGML_NUMA_MASK_LONGS];
#endif

bool ggml_numa_thread_interleave_begin(void) {
#if defined(__linux__)
    if (!ggml_numa_interleave_enabled() || g_numa_old_mode >= 0) {
        return false;
    }
    int old_mode = GGML_MPOL_DEFAULT;
    memset(g_numa_old_mask, 0, sizeof(g_numa_old_mask));
    if (syscall(SYS_get_mempolicy, &old_mode, g_numa_old_mask, GGML_NUMA_MAXNODE, NULL, 0) != 0) {
        old_mode = GGML_MPOL_DEFAULT;
    }
    if (syscall(SYS_set_mempolicy, GGML_MPOL_INTERLEAVE, g_numa_mask, GGML_NUMA_MAXNODE) != 0) {
        GGML_LOG_WARN("%s: set_mempolicy failed: %s\n", __func__, strerror(errno));
        return false;
    }
    g_numa_old_mode = old_mode;
    return true;
#else
    return false;
#endif
}

void ggml_numa_thread_interleave_end(void) {
#if defined(__linux__)
    if (g_numa_old_mode < 0) {
        return;
    }
    if (g_numa_old_mode == GGML_MPOL_DEFAULT) {
        syscall(SYS_set_mempolicy, GGML_MPOL_DEFAULT, NULL, 0);
    } else {
        syscall(SYS_set_mempolicy, g_numa_old_mode, g_numa_old_mask, GGML_NUMA_MAXNODE);
    }
    g_numa_old_mode = -1;
#endif
}

bool ggml_numa_interleave_mapped(void * addr, size_t size, int fd, size_t offset) {
#if defined(__linux__)
    if (addr == NULL || size == 0 || !ggml_numa_interleave_enabled()) {
        return false;
    }
    const size_t page = ggml_numa_page_size();
    const size_t head = (uintptr_t) addr & (page - 1);
    if (offset < head) {
        return false;
    }
    addr    = (char *) addr - head;
    offset -= head;
    size    = (size + head + page - 1) & ~(page - 1);

    // page-cache pages are placed by the faulting thread's policy, not by the mapping's (mbind has no effect on them)
    if (!ggml_numa_thread_interleave_begin()) {
        return false;
    }

    ggml_numa_populate(addr, size, page);

    // pages that were already cached kept their node: if most of them sit on one node, read them again
    const int n_nodes = ggml_numa_n_nodes();
    const double share = ggml_numa_busiest_share(addr, size, page);
    if (share > 1.0/n_nodes + 0.15) {
        GGML_LOG_INFO("%s: %.2f MiB of cached pages were %.0f%% on one node, reading them again interleaved\n",
            __func__, size/1024.0/1024.0, 100.0*share);
        madvise(addr, size, MADV_DONTNEED);
        posix_fadvise(fd, (off_t) offset, (off_t) size, POSIX_FADV_DONTNEED);
        ggml_numa_populate(addr, size, page);
    }

    ggml_numa_thread_interleave_end();
    return true;
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(size);
    GGML_UNUSED(fd);
    GGML_UNUSED(offset);
    return false;
#endif
}
