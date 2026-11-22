// CPU expert disk tier: the MoE experts of a model larger than RAM, read from the model file
// with O_DIRECT into an LRU arena of host memory, so the experts live in one bounded, owned
// cache instead of the kernel page cache. Same tier as the CUDA provider's (moe-cache.cu),
// for a machine with no accelerator: the CPU reads each expert's weights from its arena slot.
//
// The model files behind the expert tensors are found in /proc/self/maps (llama.cpp maps the
// .gguf with mmap): an expert's slice of its tensor maps to a file offset, read with pread
// into a page-aligned slot. Every failure falls back to the tensor's own address: compute then
// reads the mapping through the page cache, as it did before this tier.
//
// Environment: GGML_CPU_MOE_DISK=0 disables the tier; GGML_CPU_MOE_ARENA_MB sizes the arena
// (0 disables; default: 3/4 of the memory); GGML_CPU_MOE_DISK_STATS=1 logs the counters.

#include "moe-disk.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include "ggml-backend-moe-cache.h"

#include "moe-arena.h"

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct moe_disk_mapping {
    uintptr_t start = 0, end = 0;   // the mapped span
    uint64_t offset = 0;            // the file offset of start
    int fd = -1;                    // opened O_DIRECT, or plain when the file system has none
};

using expert_arena = moe_arena<uintptr_t, std::hash<uintptr_t>>;

struct moe_disk_shard {
    size_t slice_bytes = 0;   // one expert slice of one tensor (nb[2] of the MUL_MAT_ID weight)
    size_t slot_bytes = 0;
    expert_arena arena;
};

struct moe_disk_state {
    std::mutex mu;
    size_t budget_bytes = 0;
    bool budget_read = false;
    char * reservation = nullptr;      // the shards' memory, one mmap
    size_t reservation_bytes = 0;
    std::vector<moe_disk_mapping> maps;   // sorted by start, merged per file
    std::unordered_map<std::string, int> fd_cache;
    std::vector<std::unique_ptr<moe_disk_shard>> shards;   // stable: a fill reads its shard without the lock
    std::vector<std::pair<expert_arena *, int>> pinned;   // this node's slices, by slot
    // counters
    uint64_t n_ensure = 0, n_hit = 0, n_fill = 0, n_fallback = 0;
    uint64_t fill_us = 0, fill_us_max = 0;
    int64_t t_report = 0;

    const moe_disk_mapping * find_mapping(uintptr_t a, size_t n) {
#if defined(__linux__)
        for (int attempt = 0; attempt < 2; attempt++) {
            if (maps.empty() && !read_maps()) {
                return nullptr;
            }
            auto it = std::upper_bound(maps.begin(), maps.end(), a,
                    [](uintptr_t x, const moe_disk_mapping & m) { return x < m.start; });
            if (it != maps.begin()) {
                const moe_disk_mapping & m = *--it;
                if (a >= m.start && a + n <= m.end) {
                    return &m;
                }
            }
            // a model mapped after the last read: read the table again, once
            if (attempt == 0 && !read_maps()) {
                break;
            }
        }
#endif
        return nullptr;
    }

    bool read_maps() {
#if defined(__linux__)
        static int64_t last = INT64_MIN;
        const int64_t now = ggml_time_us();
        if (now - last < 1000000) {   // a failing lookup retries at most once a second
            return !maps.empty();
        }
        last = now;
        FILE * f = fopen("/proc/self/maps", "r");
        if (!f) {
            return false;
        }
        struct entry { uintptr_t start, end; uint64_t offset; std::string path; };
        std::vector<entry> entries;
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            unsigned long long start = 0, end = 0, offset = 0;
            char path[4096] = {};
            // the leading space skips the blanks before the path; [^\n] would take them in
            if (sscanf(line, "%llx-%llx %*s %llx %*s %*s  %4095[^\n]", &start, &end, &offset, path) < 4 ||
                    path[0] != '/') {
                continue;
            }
            entries.push_back({(uintptr_t) start, (uintptr_t) end, offset, path});
        }
        fclose(f);
        if (entries.empty()) {
            return false;
        }
        std::sort(entries.begin(), entries.end(),
                [](const entry & a, const entry & b) { return a.start < b.start; });
        std::vector<moe_disk_mapping> merged;
        for (const entry & e : entries) {
            // one file shows as several mappings where an madvise split it (the experts are
            // left out of the loader's prefetch): merged, or an expert across a split would
            // be in none of them
            auto it = fd_cache.find(e.path);
            const bool same = !merged.empty() && it != fd_cache.end() &&
                it->second >= 0 && it->second == merged.back().fd &&
                merged.back().end == e.start &&
                merged.back().offset + (merged.back().end - merged.back().start) == e.offset;
            if (!same) {
                if (it == fd_cache.end()) {
                    int fd = open(e.path.c_str(), O_RDONLY | O_DIRECT);
                    if (fd < 0) {
                        fd = open(e.path.c_str(), O_RDONLY);   // buffered if no O_DIRECT
                    }
                    it = fd_cache.emplace(e.path, fd).first;
                }
                merged.push_back({e.start, e.end, e.offset, it->second});
            } else {
                merged.back().end = e.end;
            }
        }
        maps = std::move(merged);
        return !maps.empty();
#else
        return false;
#endif
    }

    moe_disk_shard * shard_for(size_t slice_bytes) {
        for (const auto & s : shards) {
            if (s->slice_bytes == slice_bytes) {
                return s.get();
            }
        }
        if (!budget_read) {
            budget_read = true;
            const char * mb = getenv("GGML_CPU_MOE_ARENA_MB");
            budget_bytes = mb ? (size_t) atoll(mb) << 20 : auto_budget();
        }
        if (!budget_bytes) {
            return nullptr;
        }
        // a new slice size: every shard re-created with an equal share (a few, at warmup)
        std::vector<size_t> slices;
        for (const auto & s : shards) {
            slices.push_back(s->slice_bytes);
        }
        slices.push_back(slice_bytes);
        const size_t share = budget_bytes / slices.size();
        const size_t new_slot = GGML_PAD(slice_bytes + 4096, 4096);
        if (share < new_slot) {
            return nullptr;   // no slot of this size fits: the shards there are stay
        }
        size_t total = 0;
        for (const size_t bytes : slices) {
            const size_t slot = GGML_PAD(bytes + 4096, 4096);
            total += share / slot * slot;
        }
        char * base = (char *) mmap(nullptr, total, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (base == MAP_FAILED) {
            GGML_LOG_WARN("moe-disk: mmap of %zu MiB for the arena failed (%s), "
                    "the experts stay in the page cache\n", total >> 20, strerror(errno));
            return nullptr;
        }
        if (reservation) {
            munmap(reservation, reservation_bytes);   // the re-created shards' old memory
        }
        reservation = base;
        reservation_bytes = total;
        shards.clear();
        for (const size_t bytes : slices) {
            auto s = std::make_unique<moe_disk_shard>();
            s->slice_bytes = bytes;
            s->slot_bytes = GGML_PAD(bytes + 4096, 4096);
            s->arena.init(base, s->slot_bytes, (int) (share / s->slot_bytes));
            base += share / s->slot_bytes * s->slot_bytes;
            shards.push_back(std::move(s));
        }
        return shards.back().get();
    }

    static size_t auto_budget() {
#if defined(__linux__)
        FILE * f = fopen("/proc/meminfo", "r");
        if (!f) {
            return 0;
        }
        size_t total = 0;
        char name[64];
        unsigned long long kb = 0;
        while (fscanf(f, "%63s %llu", name, &kb) == 2) {
            if (strcmp(name, "MemTotal:") == 0) {
                total = (size_t) kb << 10;
                break;
            }
        }
        fclose(f);
        // the rest: the dense weights, the KV, the compute buffers, the OS
        return total ? total * 3 / 4 : 0;
#else
        return 0;
#endif
    }

    // the slice read from the file with O_DIRECT into slot (taken): the page-aligned span
    // around it, the slice at slot + lead. False on any failure (the slot is then aborted).
    bool fill(const moe_disk_mapping * m, uintptr_t a, size_t n, char * slot, size_t slot_bytes,
              const char ** part) {
#if defined(__linux__)
        const uint64_t off = m->offset + (uint64_t) (a - m->start);
        const uint64_t aligned = off & ~(uint64_t) 4095;
        const size_t lead = (size_t) (off - aligned);
        const size_t len = GGML_PAD(lead + n, 4096);
        if (len > slot_bytes) {
            return false;
        }
        size_t got = 0;
        while (got < len) {
            const ssize_t k = pread(m->fd, slot + got, len - got, (off_t) (aligned + got));
            if (k < 0 && errno == EINTR) {
                continue;
            }
            if (k <= 0) {
                break;   // the end of the file: the span's tail is past it
            }
            got += (size_t) k;
        }
        if (got < lead + n) {
            return false;
        }
        *part = slot + lead;
        return true;
#else
        GGML_UNUSED(m); GGML_UNUSED(a); GGML_UNUSED(n); GGML_UNUSED(slot); GGML_UNUSED(slot_bytes);
        GGML_UNUSED(part);
        return false;
#endif
    }

    void report() {
        if (!getenv("GGML_CPU_MOE_DISK_STATS")) {
            return;
        }
        const int64_t now = ggml_time_us();
        if (t_report && now - t_report < 5000000) {   // every 5 s
            return;
        }
        t_report = now;
        for (const auto & s : shards) {
            // stderr, not the ggml log: the tools silence INFO logs, these counters are
            // exactly what GGML_CPU_MOE_DISK_STATS asked for
            fprintf(stderr, "moe-disk: slot %zu KiB x %d: %llu hits, %llu fills, %llu evictions, "
                    "%llu fallbacks, fill avg %.2f ms max %.2f ms\n",
                    s->slot_bytes >> 10, s->arena.n_slots,
                    (unsigned long long) s->arena.hits, (unsigned long long) s->arena.writes,
                    (unsigned long long) s->arena.evictions, (unsigned long long) n_fallback,
                    n_fill ? 1e-3 * fill_us / n_fill : 0.0, 1e-3 * fill_us_max);
        }
    }
};

moe_disk_state g_state;

bool weight_eligible(const struct ggml_tensor * w) {
    if (!w || w->op != GGML_OP_NONE || !w->data || w->ne[2] <= 1 ||
            w->nb[0] != ggml_type_size(w->type) || w->nb[2] < ((size_t) 128 << 10)) {
        return false;
    }
    // the file-backed check is at ensure time (find_mapping): a weight whose bytes are not
    // the model file's (a repacked copy in the heap, another layout) finds no mapping there
    // and falls back to its own address. The host check is at the buffer type level: the
    // from_ptr buffer iface that mmap'd weights use has no is_host of its own
    const ggml_backend_buffer_t buffer = w->view_src ? w->view_src->buffer : w->buffer;
    return buffer && ggml_backend_buft_is_host(buffer->buft) &&
        ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

} // namespace

int ggml_cpu_moe_disk_enabled() {
    static const int enabled = getenv("GGML_CPU_MOE_DISK")
        ? atoi(getenv("GGML_CPU_MOE_DISK")) : 1;
    return enabled;
}

int ggml_cpu_moe_disk_wants(const struct ggml_tensor * w) {
    if (!ggml_cpu_moe_disk_enabled() || !weight_eligible(w)) {
        return 0;
    }
    // the CUDA provider hosts its own arena for these weights: one tier, not two
    if (ggml_moe_cache.begin) {
        return 0;
    }
    if (getenv("GGML_CPU_MOE_DISK_STATS")) {
        static const bool announced = (fprintf(stderr, "moe-disk: tier active for %s (%s x%d experts, %lld KiB slices)\n",
            w->name ? w->name : "?", ggml_type_name(w->type), (int) w->ne[2],
            (long long) (w->nb[2] >> 10)), true);
        GGML_UNUSED(announced);
    }
    return 1;
}

const void * ggml_cpu_moe_disk_ensure(const struct ggml_tensor * w, int64_t expert) {
    moe_disk_state & g = g_state;
    const uintptr_t addr = (uintptr_t) w->data + (size_t) expert * w->nb[2];
    const size_t slice_bytes = w->nb[2];
    char * slot_mem = nullptr;

    moe_disk_shard * s = nullptr;
    const moe_disk_mapping * m = nullptr;
    int slot = -1;
    {
        std::lock_guard<std::mutex> lock(g.mu);
        g.n_ensure++;

        s = g.shard_for(slice_bytes);
        if (!s) {
            g.n_fallback++;
            return (const void *) addr;
        }

        slot = s->arena.acquire(addr, true);   // a hit: its reader holds the slot
        if (slot >= 0) {
            g.n_hit++;
            const char * part = s->arena.slots[slot].part[0];
            g.pinned.emplace_back(&s->arena, slot);
            g.report();
            return part;
        }

        m = g.find_mapping(addr, slice_bytes);
        if (!m || m->fd < 0) {
            g.n_fallback++;
            g.report();
            return (const void *) addr;
        }

        slot = s->arena.take(addr, 0);
        if (slot < 0) {
            // another thread fills it, or the arena is full and every victim in use
            slot = s->arena.acquire(addr, true);
            if (slot < 0) {
                g.n_fallback++;
                g.report();
                return (const void *) addr;
            }
            g.n_hit++;
            const char * part = s->arena.slots[slot].part[0];
            g.pinned.emplace_back(&s->arena, slot);
            g.report();
            return part;
        }
        slot_mem = s->arena.ptr(slot);
    }

    // the read without the lock: the threads of a node fill their experts in parallel
    const char * part = nullptr;
    const int64_t t0 = ggml_time_us();
    const bool ok = g.fill(m, addr, slice_bytes, slot_mem, s->slot_bytes, &part);
    const int64_t dt = ggml_time_us() - t0;

    {
        std::lock_guard<std::mutex> lock(g.mu);
        g.fill_us += (uint64_t) dt;
        g.fill_us_max = std::max(g.fill_us_max, (uint64_t) dt);
        if (!ok) {
            s->arena.abort(slot);
            g.n_fallback++;
            g.report();
            return (const void *) addr;
        }
        s->arena.slots[slot].part[0] = part;
        s->arena.commit(slot);
        s->arena.acquire(addr, true);   // pinned for the node, as a hit's
        g.n_fill++;
        g.pinned.emplace_back(&s->arena, slot);
        g.report();
        return part;
    }
}

void ggml_cpu_moe_disk_node_end() {
    moe_disk_state & g = g_state;
    std::lock_guard<std::mutex> lock(g.mu);
    for (const auto & p : g.pinned) {
        p.first->release(p.second);
    }
    g.pinned.clear();
}
