#pragma once

#include "utils/gpu_random.h"
#include "gpu_truncate.h"
#include "gpu_dpf.h"

template <typename T>
struct GpuFssF2BF16Key
{
    int N, bin;

    // DCF key written by gpuKeyGenDCF — read by gpuDcf inside evaluator
    // This is the same GPUDPFKey format used everywhere else in the codebase
    GPUDPFKey dcfKey;

    // Beaver blinding factors (CPU pointers, N elements each)
    T *rout_k;
    T *rout_m;
    T *prod;
    T *rin;

    // DaBit material:
    //   r_xor[i*N + j]        = XOR share for element j at bit position i
    //   r_arithmetic[i*N + j] = arithmetic share for element j at bit position i
    // Both sized bin * N
    u8 *r_xor;
    T  *r_arithmetic;

    // Truncation key for res2 (GPU truncation format)
    // NOTE: see the res2 fix note — if you apply the direct-shift fix,
    // this field and dcfTruncate keygen below can be removed entirely.
    GPUTruncateKey<T> dcfTruncate;
};

template <typename T>
GpuFssF2BF16Key<T> readGpuFssF2BF16Key(u8 **key_as_bytes)
{
    GpuFssF2BF16Key<T> k;

    // l.bout = (int)**key_as_bytes;
    // *key_as_bytes += sizeof(int);

    // k.N   = (int)**key_as_bytes;
    // *key_as_bytes += sizeof(int);
    // k.bin = (int)**key_as_bytes;
    // *key_as_bytes += sizeof(int);

    memcpy(&k, *key_as_bytes, 2 * sizeof(int));
    *key_as_bytes += 2 * sizeof(int);

    // GPU DCF key (written by gpuKeyGenDCF — use the existing GPU DCF reader)
    k.dcfKey = readGPUDPFKey(key_as_bytes);

    // Truncation key (CPU format, read by readCpuFssTruncateKey)
    // Remove this if you apply the direct-shift fix in the evaluator.
    k.dcfTruncate = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);

    // Beaver factors: N elements each, stored as additive shares
    u64 Nsz = k.N * sizeof(T);
    k.rout_k = (T *)*key_as_bytes;  *key_as_bytes += Nsz;
    k.rout_m = (T *)*key_as_bytes;  *key_as_bytes += Nsz;
    k.prod   = (T *)*key_as_bytes;  *key_as_bytes += Nsz;
    k.rin    = (T *)*key_as_bytes;  *key_as_bytes += Nsz;

    // DaBit material: bin*N bytes for r_xor, bin*N T-elements for r_arithmetic
    u64 dabit_xor_sz   = k.bin * k.N * sizeof(u8);
    u64 dabit_arith_sz = k.bin * k.N * sizeof(T);
    k.r_xor        = (u8 *)*key_as_bytes;  *key_as_bytes += dabit_xor_sz;
    k.r_arithmetic = (T  *)*key_as_bytes;  *key_as_bytes += dabit_arith_sz;

    return k;
}

#include "gpu_f2bf16.cu"