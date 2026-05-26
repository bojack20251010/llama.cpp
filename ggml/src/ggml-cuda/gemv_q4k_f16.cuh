#pragma once
// Q4_K GEMV variants (no dp4a, no tensor core):
//   Q4K_GEMV_MODE=f16   → fp16 coeff-precompute + HMUL2    (for CMP fp16-full)
//   Q4K_GEMV_MODE=f32   → pure f32 fmaf + float4 act load  (for CMP fp32-full)
//   Q4K_FP16_GEMV=1     → default = f16 mode
// Enable: Q4K_FP16_GEMV=1

static __device__ __forceinline__ float _hsum2(const half2 & h) { return __half2float(h.x) + __half2float(h.y); }

// ── f32 pure: coeff = ws*w - wm (f32), float4 act load, fmaf accumulate ──
static __global__ void gemv_q4_K_f32_k(
    const block_q4_K * __restrict__ w,
    const float       * __restrict__ x,
    float             * __restrict__ y,
    const int K, const int N) {

    const int row = blockIdx.x;
    if (row >= N) return;
    const int n_blocks = K / QK_K;
    const block_q4_K * w_row = w + (size_t)row * n_blocks;
    float acc = 0.0f;

    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        const block_q4_K * blk = &w_row[b];
        const float d = __half2float(blk->dm.x), dmin = __half2float(blk->dm.y);
        const uint8_t * sq = blk->scales;

        float sub_sc[8], sub_m[8];
        for (int j = 0; j < 8; j++) {
            if (j < 4) { sub_sc[j] = d * (sq[j] & 63);      sub_m[j] = dmin * (sq[j+4] & 63); }
            else       { sub_sc[j] = d * ((sq[j+4]&0xF)|((sq[j-4]>>6)<<4));
                         sub_m[j]  = dmin * ((sq[j+4]>>4)|((sq[j-0]>>6)<<4)); }
        }

#pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const float ws = sub_sc[sub], wm = sub_m[sub];
            const bool use_high = (sub & 1);
            const int q_base = (sub / 2) * 32;

#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const int q_off = q_base + j;
                const int a_off = b * QK_K + sub * 32 + j;

                uint32_t qv; memcpy(&qv, blk->qs + q_off, 4);
                float w0, w1, w2, w3;
                if (use_high) { w0=(float)((qv>> 4)&0xF); w1=(float)((qv>>12)&0xF);
                                w2=(float)((qv>>20)&0xF); w3=(float)((qv>>28)&0xF); }
                else          { w0=(float)(qv & 0xF);      w1=(float)((qv>> 8)&0xF);
                                w2=(float)((qv>>16)&0xF);  w3=(float)((qv>>24)&0xF); }

                const float4 xv = *(const float4*)(x + a_off);

                acc = fmaf(ws*w0-wm, xv.x, fmaf(ws*w1-wm, xv.y,
                           fmaf(ws*w2-wm, xv.z, fmaf(ws*w3-wm, xv.w, acc))));
            }
        }
    }
    for (int off=16;off>0;off>>=1) acc+=__shfl_xor_sync(0xffffffff,acc,off);
    int wid=threadIdx.x>>5, lane=threadIdx.x&31, nw=blockDim.x>>5;
    __shared__ float sm[8]; if(lane==0) sm[wid]=acc; __syncthreads();
    if(wid==0){float v=(lane<nw)?sm[lane]:0.0f; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(lane==0)y[row]=v;}
}

// ── fp16 coeff-precompute: scale calc in half domain, I2H for nibbles, HFMA2, HADD2 ──
static __global__ void gemv_q4_K_f16_k(
    const block_q4_K * __restrict__ w,
    const float       * __restrict__ x,
    float             * __restrict__ y,
    const int K, const int N) {

    const int row = blockIdx.x;
    if (row >= N) return;
    const int n_blocks = K / QK_K;
    const block_q4_K * w_row = w + (size_t)row * n_blocks;
    float acc = 0.0f;

    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        const block_q4_K * blk = &w_row[b];
        const half d_h = blk->dm.x, dmin_h = blk->dm.y;
        const uint8_t * sq = blk->scales;

        half sub_sc_h[8], sub_m_h[8];
        for (int j = 0; j < 8; j++) {
            if (j < 4) {
                sub_sc_h[j] = __hmul(d_h,    __int2half_rn(sq[j] & 63));
                sub_m_h[j]  = __hmul(dmin_h, __int2half_rn(sq[j+4] & 63));
            } else {
                sub_sc_h[j] = __hmul(d_h,    __int2half_rn((sq[j+4]&0xF)|((sq[j-4]>>6)<<4)));
                sub_m_h[j]  = __hmul(dmin_h, __int2half_rn((sq[j+4]>>4)|((sq[j-0]>>6)<<4)));
            }
        }

#pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const half ws_h = sub_sc_h[sub], wm_h = sub_m_h[sub];
            const half2 ws_h2  = __halves2half2(ws_h, ws_h);
            const half2 nwm_h2 = __halves2half2(__hneg(wm_h), __hneg(wm_h));
            const bool use_high = (sub & 1);
            const int q_base = (sub / 2) * 32;

#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const int q_off = q_base + j;
                const int a_off = b * QK_K + sub * 32 + j;

                uint32_t qv; memcpy(&qv, blk->qs + q_off, 4);
                half w0, w1, w2, w3;
                if (use_high) { w0=__int2half_rn((qv>> 4)&0xF); w1=__int2half_rn((qv>>12)&0xF);
                                w2=__int2half_rn((qv>>20)&0xF); w3=__int2half_rn((qv>>28)&0xF); }
                else          { w0=__int2half_rn(qv & 0xF);      w1=__int2half_rn((qv>> 8)&0xF);
                                w2=__int2half_rn((qv>>16)&0xF);  w3=__int2half_rn((qv>>24)&0xF); }

                half2 c01 = __hfma2(ws_h2, __halves2half2(w0,w1), nwm_h2);
                half2 c23 = __hfma2(ws_h2, __halves2half2(w2,w3), nwm_h2);

                const float4 xv = *(const float4*)(x + a_off);
                half2 x01 = __floats2half2_rn(xv.x, xv.y);
                half2 x23 = __floats2half2_rn(xv.z, xv.w);

                acc += _hsum2(__hadd2(__hmul2(c01, x01), __hmul2(c23, x23)));
            }
        }
    }
    for (int off=16;off>0;off>>=1) acc+=__shfl_xor_sync(0xffffffff,acc,off);
    int wid=threadIdx.x>>5, lane=threadIdx.x&31, nw=blockDim.x>>5;
    __shared__ float sm[8]; if(lane==0) sm[wid]=acc; __syncthreads();
    if(wid==0){float v=(lane<nw)?sm[lane]:0.0f; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(lane==0)y[row]=v;}
}

