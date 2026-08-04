#pragma once

#include "gpu_f2bf16.h"
#include "gpu_lut.h"

// ============================================================
// Key struct
// ============================================================
template <typename T>
struct GpuFssRsqrtKey
{
    GpuFssF2BF16Key<T> f2bf16Key;
    GPULUTKey<T>       lutKey;
};

// ============================================================
// Key reader
// Reads exactly what gpuFssKeyGenRsqrt writes:
//   [F2BF16 key][GPU LUT key]
// ============================================================
template <typename T>
GpuFssRsqrtKey<T> readGpuFssRsqrtKey(u8 **key_as_bytes)
{
    GpuFssRsqrtKey<T> k;
    k.f2bf16Key = readGpuFssF2BF16Key<T>(key_as_bytes);
    k.lutKey    = readGPULUTKey<T>(key_as_bytes);
    return k;
}

#include "gpu_rsqrt.cu"