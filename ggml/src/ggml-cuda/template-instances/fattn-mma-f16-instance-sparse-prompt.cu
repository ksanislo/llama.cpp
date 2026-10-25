// The 16-head tiles of the 256/256 sparse prompt attention (GGML_CUDA_FA_SPARSE_PROMPT, fattn.cu): gqa 12 in one tile.
// Not generated: generate_cu_files.py compiles ncols2 = 16 only for the head sizes that use it by default.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE(256, 256, 1, 16);
DECL_FATTN_MMA_F16_CASE(256, 256, 2, 16);
