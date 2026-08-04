#pragma once

#include "gpu_f2bf16.h"

// ============================================================
// Keygen kernel: prod[i] = rinArr[i] * rout_m[i] + rout_xm[i]
// ============================================================
template <typename T>
__global__ void f2bf16KeygenProd(int N, T *d_rin, T *d_rout_m, T *d_rout_xm, T *d_prod)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        d_prod[i] = d_rin[i] * d_rout_m[i] + d_rout_xm[i];
}

// ============================================================
// Keygen kernel: DaBit generation
// For each index i in [0, total_iter):
//   d_r_xor[i]        = party==SERVER1 ? (r_clear^r_x0) : r_x0
//   d_r_arithmetic[i] = party==SERVER1 ? (r_clear - r_a0) : r_a0
// ============================================================
template <typename T>
__global__ void f2bf16KeygenDaBit(int party, int total_iter,
                                   u8 *d_r_clear, u8 *d_r_x0, T *d_r_a0,
                                   u8 *d_r_xor, T *d_r_arithmetic)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_iter) return;

    u8 rc = d_r_clear[i] & 1;
    u8 rx = d_r_x0[i]    & 1;
    T  ra = d_r_a0[i];

    d_r_xor[i]        = (party == SERVER1) ? (rc ^ rx) : rx;
    d_r_arithmetic[i] = (party == SERVER1) ? (T(rc) - ra) : ra;
}

// ============================================================
// Keygen kernel: pack final 13-bit output mask
// final_mask[i] = (rout_k[i] mod 64) + (d_truncateMask[i] << 6) mod 2^13
// ============================================================
template <typename T>
__global__ void f2bf16KeygenPackMask(int N, T *d_rout_k, T *d_truncateMask, T *d_final_mask)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    T k = d_rout_k[i];
    gpuMod(k, 6);
    T val = k + (d_truncateMask[i] << 6);
    gpuMod(val, 13);
    d_final_mask[i] = val;
}

// ============================================================
// Eval kernel 1: compute y_array
// y[(i-1)*N + j] = h_x[j] - 2^i  (mod bin),  i in [1, bin-1]
// ============================================================
template <typename T>
__global__ void f2bf16ComputeYArray(int bin, int N, T *d_x, T *d_y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (bin - 1) * N) return;
    int i = tid / N + 1;
    int j = tid % N;
    T c = T(1) << i;
    d_y[tid] = d_x[j] - c;
    gpuMod(d_y[tid], bin);
}

// ============================================================
// Eval kernel 2: XOR DCF bits with r_xor → packed e_complete
// e_complete layout: bin slices of packed_u8s bytes each.
//   slot 0      = t1 (DCF on zeros)
//   slot 1..bin-1 = t2_i (DCF on y chunk i)
// d_t_all: bin * dcf_out_u32s  u32s, where dcf_out_u32s =
//          (N - 1)/32 + 1 words per DCF call, laid out as
//          [t1 words | t2_1 words | ... | t2_{bin-1} words]
// ============================================================
__global__ void f2bf16PackAndXor(
    int bin, int N, int packed_u8s,
    u32 **d_t_ptrs,     // bin device pointers to DCF outputs
    u8   *d_r_xor,      // bin * N bytes
    u8   *d_e_complete) // bin * packed_u8s bytes output
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bin * packed_u8s) return;
    int slot     = tid / packed_u8s;
    int byte_idx = tid % packed_u8s;
    u32 *slot_t  = d_t_ptrs[slot];
    u8 result = 0;
    for (int bit_idx = 0; bit_idx < 8; bit_idx++)
    {
        int l = byte_idx * 8 + bit_idx;
        if (l >= N) break;
        u8 dcf_bit = (slot_t[l / 32] >> (l % 32)) & 1;
        u8 r       = d_r_xor[slot * N + l] & 1;
        if (dcf_bit ^ r) result |= (1 << bit_idx);
    }
    d_e_complete[tid] = result;
}

// ============================================================
// Eval kernel 3: reconstructed bits → arithmetic shares + boundary
// → res_values[i*N + j]
// ============================================================
template <typename T>
__global__ void f2bf16ComputeResValues(
    int party, int bin, int N, int packed_u8s,
    u8 *d_e_complete, T *d_r_arithmetic, T *d_y, T *d_res_values)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bin * N) return;

    int i = tid / N;
    int j = tid % N;

    u8 byte_val = d_e_complete[i * packed_u8s + j / 8];
    int bit     = (byte_val >> (j % 8)) & 1;

    T arith;
    if (bit == 0)
        arith = d_r_arithmetic[i * N + j];
    else
        arith = (party == SERVER1) ? (T(0) - d_r_arithmetic[i * N + j])
                                   : (T(1) - d_r_arithmetic[i * N + j]);

    // Boundary correction: only SERVER1, only i >= 1
    if (party == SERVER1 && i >= 1)
    {
        T c     = T(1) << i;
        T N_val = -c;
        gpuMod(N_val, bin);
        if (d_y[(i - 1) * N + j] >= N_val)
            arith += T(1);
    }

    d_res_values[tid] = arith;
}

// ============================================================
// Eval kernel 4: accumulate res_values → k_final, m_final
// One thread per element j; iterates over all i sequentially.
// bin=50 inner iterations — fast enough without further parallelism.
// ============================================================
template <typename T>
__global__ void f2bf16Accumulate(
    int bin, int N, T *d_res_values,
    T *d_rout_k, T *d_rout_m,
    T *d_k_final, T *d_m_final)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    T k_acc = 0, m_acc = 0;
    for (int i = 1; i < bin; i++)
    {
        T delta = d_res_values[i * N + j] - d_res_values[(i - 1) * N + j];
        k_acc += T(i - 1) * delta;
        m_acc += (T(1) << (bin - i)) * delta;
    }
    d_k_final[j] = k_acc + d_rout_k[j];
    d_m_final[j] = m_acc + d_rout_m[j];
}

// ============================================================
// Eval kernel 5: Beaver → res2
// ============================================================
template <typename T>
__global__ void f2bf16ComputeRes2(
    int party, int N,
    T *d_rin, T *d_rout_m, T *d_prod, T *d_hx, T *d_m_final, T *d_res2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    T val = -d_rin[j] * d_m_final[j]
            - d_hx[j] * d_rout_m[j]
            + d_prod[j];
    if (party == SERVER1)
        val += d_m_final[j] * d_hx[j];
    d_res2[j] = val;
}

// ============================================================
// Eval kernel 6: pack 13-bit LUT index
// t3 is already public (after reconstructInPlace on res2).
// Direct right-shift — no MPC truncation needed.
// index = ((public_res2 >> (bin-8)) - 128) mod 8) << 6 | (k_final mod 64)
// ============================================================
template <typename T>
__global__ void f2bf16PackIndex(int party, int bin, int N,
                                 T *d_t3, T *d_k_final, T *d_out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    T mantissa = d_t3[j] - 128;   // both parties, no guard — matches CPU
    gpuMod(mantissa, 8);

    T k = d_k_final[j];
    gpuMod(k, 6);

    d_out[j] = (mantissa << 6) | k;
    gpuMod(d_out[j], 13);
}


// ============================================================
// GPU F2BF16 Keygen
// ============================================================
template <typename T>
T *gpuFssKeyGenF2BF16(u8 **key_as_bytes, int party, int bin, int N,
                       T *d_rinArr,           // GPU pointer, N elements
                       AESGlobalContext *gaes)
{
    writeInt(key_as_bytes, N);
    writeInt(key_as_bytes, bin);

    // DCF key — d_rinArr stays alive through keygen
    gpuKeyGenDCF<T>(key_as_bytes, party, bin, N, d_rinArr, gaes);

    // Beaver blinding factors (all on GPU)
    T *d_rout_k  = randomGEOnGpu<T>(N, bin);
    T *d_rout_m  = randomGEOnGpu<T>(N, bin);
    T *d_rout_xm = randomGEOnGpu<T>(N, bin);
    T *d_prod    = (T *)gpuMalloc(N * sizeof(T));

    f2bf16KeygenProd<<<(N-1)/256+1, 256>>>(N, d_rinArr, d_rout_m, d_rout_xm, d_prod);
    checkCudaErrors(cudaDeviceSynchronize());

    // Truncation key for rout_xm (shift = bin-8)
    // NOTE: the evaluator uses a direct right-shift on public res2,
    // so this key is only needed if you keep the StTrunc3R path.
    // Remove this block if you apply the direct-shift fix end-to-end.
    auto d_truncateMask = genGPUTruncateKey<T, T>(
        key_as_bytes, party, TruncateType::TrWithSlack,
        bin, 8, bin - 8, N, d_rout_xm, gaes);

    // Write Beaver shares (writeShares expects GPU pointers)
    writeShares<T, T>(key_as_bytes, party, N, d_rout_k,  bin, true);
    writeShares<T, T>(key_as_bytes, party, N, d_rout_m,  bin, true);
    writeShares<T, T>(key_as_bytes, party, N, d_prod,    bin, true);
    writeShares<T, T>(key_as_bytes, party, N, d_rinArr,  bin, true);

    gpuFree(d_rout_xm);
    gpuFree(d_prod);

    // DaBit generation (all on GPU)
    int total_iter    = bin * N;
    u8 *d_r_xor       = (u8 *)gpuMalloc(total_iter * sizeof(u8));
    T  *d_r_arithmetic = (T  *)gpuMalloc(total_iter * sizeof(T));

    u8 *d_r_clear = randomGEOnGpu<u8>(total_iter, 1);
    u8 *d_r_x0    = randomGEOnGpu<u8>(total_iter, 1);
    T  *d_r_a0    = randomGEOnGpu<T> (total_iter, sizeof(T) * 8);

    f2bf16KeygenDaBit<<<(total_iter-1)/256+1, 256>>>(
        party, total_iter, d_r_clear, d_r_x0, d_r_a0, d_r_xor, d_r_arithmetic);
    checkCudaErrors(cudaDeviceSynchronize());

    gpuFree(d_r_clear);
    gpuFree(d_r_x0);
    gpuFree(d_r_a0);

    // writeShares handles packing and CPU write
    writeShares<u8, u8>(key_as_bytes, party, total_iter, d_r_xor,        8,            false);
    writeShares<T,  T >(key_as_bytes, party, total_iter, d_r_arithmetic, sizeof(T)*8,  false);

    gpuFree(d_r_xor);
    gpuFree(d_r_arithmetic);

    // Output mask
    T *d_final_mask = (T *)gpuMalloc(N * sizeof(T));
    f2bf16KeygenPackMask<<<(N-1)/256+1, 256>>>(N, d_rout_k, d_truncateMask, d_final_mask);
    checkCudaErrors(cudaDeviceSynchronize());

    gpuFree(d_truncateMask);
    gpuFree(d_rout_k);

    return d_final_mask;  // GPU pointer — caller writes it via writeShares or moveToCPU
}


// ============================================================
// GPU F2BF16 Evaluator
// 4 network rounds:
//   1. reconstructBitsInPlace on e_complete (all DCF bits at once)
//   2. reconstructInPlace on k_final
//   3. reconstructInPlace on m_final
//   4. reconstructInPlace on res2
// DCF key uploaded once, reused for all bin calls.
// ============================================================
template <typename T>
T *gpuFssF2BF16(SigmaPeer *peer, int party, T *d_x,
                GpuFssF2BF16Key<T> &k,
                AESGlobalContext *gaes, Stats *s, int OpType = 0)
{
    const int N          = k.N;
    const int bin        = k.bin;
    const int packed_u8s = (N - 1) / 8 + 1;

    // B==1 for all realistic N (bin=50 → memSzOneK ≈ 768B → m ≈ 33M)
    assert(k.dcfKey.B == 1);
    GPUDPFTreeKey &tk = k.dcfKey.dpfTreeKey[0];

    // ---- Upload DCF key ONCE ----
    uint4 *d_scw = (uint4 *)moveToGPU((u8 *)tk.scw, tk.memSzScw, s, OpType);
    uint4 *d_l0  = (uint4 *)moveToGPU((u8 *)tk.l0,  tk.memSzL,   s, OpType);
    uint4 *d_l1  = (uint4 *)moveToGPU((u8 *)tk.l1,  tk.memSzL,   s, OpType);
    u32   *d_tR  = (u32   *)moveToGPU((u8 *)tk.tR,  tk.memSzT,   s, OpType);

    // ---- Phase 1: bin DCF evals — no network, key reused ----

    // Collect output pointers (host array of device pointers)
    u32 **h_t_ptrs = new u32*[bin];

    // Slot 0: DCF on zeros
    T *d_zeros = (T *)gpuMalloc(N * sizeof(T));
    checkCudaErrors(cudaMemset(d_zeros, 0, N * sizeof(T)));
    h_t_ptrs[0] = gpuDcf<T, 1, idPrologue, idEpilogue>(
        k.dcfKey, party, d_zeros, gaes, s, NULL, OpType,
        d_scw, d_l0, d_l1, d_tR);
    gpuFree(d_zeros);

    // Slots 1..bin-1: DCF on y chunks
    T *d_y = (T *)gpuMalloc((bin - 1) * N * sizeof(T));
    f2bf16ComputeYArray<<<((bin-1)*N-1)/256+1, 256>>>(bin, N, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 1; i < bin; i++)
    {
        h_t_ptrs[i] = gpuDcf<T, 1, idPrologue, idEpilogue>(
            k.dcfKey, party, d_y + (i - 1) * N, gaes, s, NULL, OpType,
            d_scw, d_l0, d_l1, d_tR);
    }

    // Free DCF key — all bin evals done
    gpuFree(d_scw); gpuFree(d_l0); gpuFree(d_l1); gpuFree(d_tR);

    // Upload host pointer array to device for the XOR kernel
    u32 **d_t_ptrs = (u32 **)gpuMalloc(bin * sizeof(u32 *));
    checkCudaErrors(cudaMemcpy(d_t_ptrs, h_t_ptrs,
                               bin * sizeof(u32 *), cudaMemcpyHostToDevice));
    delete[] h_t_ptrs;

    // XOR with r_xor → e_complete
    u8 *d_r_xor     = (u8 *)moveToGPU((u8 *)k.r_xor, bin * N * sizeof(u8), s, OpType);
    int total_bytes  = bin * packed_u8s;
    u8 *d_e_complete = (u8 *)gpuMalloc(total_bytes);
    checkCudaErrors(cudaMemset(d_e_complete, 0, total_bytes));
    f2bf16PackAndXor<<<(total_bytes-1)/256+1, 256>>>(
        bin, N, packed_u8s, d_t_ptrs, d_r_xor, d_e_complete);
    checkCudaErrors(cudaDeviceSynchronize());

    // Free DCF outputs (need host copies of pointers to call gpuFree)
    u32 **h_t_ptrs2 = new u32*[bin];
    checkCudaErrors(cudaMemcpy(h_t_ptrs2, d_t_ptrs,
                               bin * sizeof(u32 *), cudaMemcpyDeviceToHost));
    for (int i = 0; i < bin; i++) gpuFree(h_t_ptrs2[i]);
    delete[] h_t_ptrs2;
    gpuFree(d_t_ptrs);
    gpuFree(d_r_xor);

    // ---- NETWORK ROUND 1: reconstruct all bits ----
    peer->reconstructInPlace(d_e_complete, 1, (u64)bin * N, s, OpType);

    // ---- Phase 2: arithmetic shares + accumulation ----
    T *d_r_arith  = (T *)moveToGPU((u8 *)k.r_arithmetic, bin * N * sizeof(T), s, OpType);
    T *d_res_vals = (T *)gpuMalloc(bin * N * sizeof(T));
    checkCudaErrors(cudaMemset(d_res_vals, 0, bin * N * sizeof(T)));
    f2bf16ComputeResValues<<<(bin*N-1)/256+1, 256>>>(
        party, bin, N, packed_u8s, d_e_complete, d_r_arith, d_y, d_res_vals);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_e_complete); gpuFree(d_r_arith); gpuFree(d_y);

    T *d_rout_k  = (T *)moveToGPU((u8 *)k.rout_k, N * sizeof(T), s, OpType);
    T *d_rout_m  = (T *)moveToGPU((u8 *)k.rout_m, N * sizeof(T), s, OpType);
    T *d_k_final = (T *)gpuMalloc(N * sizeof(T));
    T *d_m_final = (T *)gpuMalloc(N * sizeof(T));
    f2bf16Accumulate<<<(N-1)/256+1, 256>>>(
        bin, N, d_res_vals, d_rout_k, d_rout_m, d_k_final, d_m_final);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_res_vals); gpuFree(d_rout_k);

    // ---- NETWORK ROUNDS 2 & 3: open k_final, m_final ----
    peer->reconstructInPlace(d_k_final, bin, (u64)N, s, OpType);
    peer->reconstructInPlace(d_m_final, bin, (u64)N, s, OpType);

    // ---- Phase 3: Beaver → res2 ----
    T *d_rin  = (T *)moveToGPU((u8 *)k.rin,  N * sizeof(T), s, OpType);
    T *d_prod = (T *)moveToGPU((u8 *)k.prod, N * sizeof(T), s, OpType);
    T *d_res2 = (T *)gpuMalloc(N * sizeof(T));
    f2bf16ComputeRes2<<<(N-1)/256+1, 256>>>(
        party, N, d_rin, d_rout_m, d_prod, d_x, d_m_final, d_res2);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_rin); gpuFree(d_prod); gpuFree(d_rout_m);

    // ---- NETWORK ROUND 4: open res2 ----
    peer->reconstructInPlace(d_res2, bin, (u64)N, s, OpType);

    // ---- Phase 4: direct right-shift on public res2 → 13-bit index ----
    // res2 is fully public — no MPC truncation needed here.
    T *d_out = (T *)gpuMalloc(N * sizeof(T));
    // Truncate res2 as a secret share (matching CPU exactly)
    T *d_t3 = gpuTruncate<T, T>(bin, 8, TruncateType::TrWithSlack,
                                k.dcfTruncate, bin - 8,
                                peer, party, N, d_res2, gaes, s, OpType);
    gpuFree(d_res2);

    // Pack index — both parties subtract 128 (no party guard, matching CPU)
    f2bf16PackIndex<<<(N-1)/256+1, 256>>>(party, bin, N, d_t3, d_k_final, d_out);
        checkCudaErrors(cudaDeviceSynchronize());
    
    gpuFree(d_k_final); 
    gpuFree(d_m_final);
    gpuFree(d_t3);

    return d_out;
}
