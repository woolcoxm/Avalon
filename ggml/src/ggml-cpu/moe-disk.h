#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

// the tier's env switch (GGML_CPU_MOE_DISK, on by default). When on, the loader must keep
// the expert weights file-mapped in the plain CPU buffer: the tier reads the file itself,
// and a repacked second copy of the experts would not fit host memory beside the arena
int ggml_cpu_moe_disk_enabled(void);

// Whether the expert disk tier may serve this MUL_MAT_ID weight tensor: a host weights
// buffer, more than one expert, and the tier enabled (GGML_CPU_MOE_DISK, on by default).
// Cheap and stable: all threads of a node get the same answer.
int ggml_cpu_moe_disk_wants(const struct ggml_tensor * w);

// One expert's weight slice (w->data + expert * w->nb[2]), resident in the arena. Reads the
// model file with O_DIRECT into an LRU slot on a miss. Returns the tensor's own address
// whenever the tier cannot serve it (not file-backed, read failed, arena full): the caller
// then computes from the mapping through the page cache, as without the tier.
// The slice stays pinned until ggml_cpu_moe_disk_node_end().
const void * ggml_cpu_moe_disk_ensure(const struct ggml_tensor * w, int64_t expert);

// Unpin the slices ggml_cpu_moe_disk_ensure() pinned for the node that ended. Call from one
// thread behind the barrier that ends the node's compute.
void ggml_cpu_moe_disk_node_end(void);

#ifdef __cplusplus
}
#endif
