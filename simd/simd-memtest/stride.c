#include "stride.h"

void stride_ref(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=s) {
        b[i] = a[i];
    }
}

#if __x86_64__
void stride_mov(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=s) {
#if __x86_64__
        double t;
        //t = a[i];
        asm ("mov %1, %0" : "=r" (t) : "m" (a[i]));
        //b[i] = t;
        asm ("mov %1, %0" : "=m" (b[i]) : "r" (t));
#elif __aarch64__
#warning unimplemented
#else
#error unsupported ISA
#endif
    }
}
#endif

#ifdef __SSE2__
void stride_movnti(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
#if __x86_64__
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=s) {
        double t;
        //t = a[i];
        asm ("mov %1, %0" : "=r" (t) : "m" (a[i]));
        //b[i] = t;
        asm ("movnti %1, %0" : "=m" (b[i]) : "r" (t));
    }
    asm ("sfence" ::: "memory");
#elif __aarch64__
#warning unimplemented
#else
#error unsupported ISA
#endif
}

#ifdef __INTEL_COMPILER
void stride_movnti64(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=s) {
        const __m64 t = _m_from_int64( *(__int64*)&(a[i]) );
        _mm_stream_si64( (__int64*)&(b[i]), *(__int64*)&t);
    }
    _mm_sfence();
}

void stride_movntq64(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
    //_mm_empty();
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=s) {
        const __m64 t = _m_from_int64( *(__int64*)&(a[i]) );
        _mm_stream_pi( (__m64*)&(b[i]), (__m64)t);
    }
    _mm_sfence();
}
#endif

#endif /* SSE2 */

#ifdef __AVX2__
void stride_vgatherdpd128(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
    const __m128i vindex = _mm_set_epi32(-1,-1,s,0); // start from the right...
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=(2*s)) {
        const __m128d t = _mm_i32gather_pd( &(a[i]), vindex, 8 /* scale */ );
        _mm_storel_pd( &(b[i]), t);
        _mm_storeh_pd( &(b[i+s]), t);
    }
}

void stride_vgatherqpd128(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
    const __m128i vindex = _mm_set_epi64x(s,0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=(2*s)) {
        const __m128d t = _mm_i64gather_pd( &(a[i]), vindex, 8 /* scale */ );
        _mm_storel_pd( &(b[i]), t);
        _mm_storeh_pd( &(b[i+s]), t);
    }
}

void stride_vgatherdpd256(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
    const __m128i vindex = _mm_set_epi32(s*3,s*2,s,0); // start from the right...
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=(4*s)) {
        const __m256d t = _mm256_i32gather_pd( &(a[i]), vindex, 8 /* scale */ );
        const __m128d l = _mm256_extractf128_pd(t,0);
        const __m128d u = _mm256_extractf128_pd(t,1);
        _mm_storel_pd( &(b[i    ]), l);
        _mm_storeh_pd( &(b[i+s  ]), l);
        _mm_storel_pd( &(b[i+s*2]), u);
        _mm_storeh_pd( &(b[i+s*3]), u);
    }
}

void stride_vgatherqpd256(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
    const __m256i vindex = _mm256_set_epi64x(s*3,s*2,s,0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=(4*s)) {
        const __m256d t = _mm256_i64gather_pd( &(a[i]), vindex, 8 /* scale */ );
        const __m128d l = _mm256_extractf128_pd(t,0);
        const __m128d u = _mm256_extractf128_pd(t,1);
        _mm_storel_pd( &(b[i    ]), l);
        _mm_storeh_pd( &(b[i+s  ]), l);
        _mm_storel_pd( &(b[i+s*2]), u);
        _mm_storeh_pd( &(b[i+s*3]), u);
    }
}
#endif /* AVX2 */

#ifdef __AVX512F__
static inline void stride3_mvmovapd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    __m512d src = {0};
    __mmask8 k0 =  73; // 10010010
    __mmask8 k1 = 146; // 01001001
    __mmask8 k2 =  36; // 00100100
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=24) {
        const __m512d t0 = _mm512_mask_load_pd( src, k0, &(a[i   ]) );
        const __m512d t1 = _mm512_mask_load_pd( src, k1, &(a[i+ 8]) );
        const __m512d t2 = _mm512_mask_load_pd( src, k2, &(a[i+16]) );
        _mm512_mask_store_pd( &(b[i   ]), k0, t0);
        _mm512_mask_store_pd( &(b[i+ 8]), k1, t1);
        _mm512_mask_store_pd( &(b[i+16]), k2, t2);
    }
}

static inline void stride5_mvmovapd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m512d src = {0};
    const __mmask8 k0 =  33; // 10000100
    const __mmask8 k1 = 132; // 00100001
    const __mmask8 k2 =  16; // 00001000
    const __mmask8 k3 =  66; // 01000010
    const __mmask8 k4 =   8; // 00001000
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=40) {
        const __m512d t0 = _mm512_mask_load_pd( src, k0, &(a[i   ]) );
        const __m512d t1 = _mm512_mask_load_pd( src, k1, &(a[i+ 8]) );
        const __m512d t2 = _mm512_mask_load_pd( src, k2, &(a[i+16]) );
        const __m512d t3 = _mm512_mask_load_pd( src, k3, &(a[i+24]) );
        const __m512d t4 = _mm512_mask_load_pd( src, k4, &(a[i+32]) );
        _mm512_mask_store_pd( &(b[i   ]), k0, t0);
        _mm512_mask_store_pd( &(b[i+ 8]), k1, t1);
        _mm512_mask_store_pd( &(b[i+16]), k2, t2);
        _mm512_mask_store_pd( &(b[i+24]), k3, t3);
        _mm512_mask_store_pd( &(b[i+32]), k4, t4);
    }
}

void stride_mvmovapd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    const __m512d src = {0};
    __mmask8 k = 255;
    switch (s) {
        case 1: k = 255; break;
        case 2: k =  85; break;
        case 3: { return stride3_mvmovapd512(n,a,b); }
        case 4: k =  17; break;
        case 5: { return stride5_mvmovapd512(n,a,b); }
        case 8: k =   1; break;
        default: { return stride_ref(n,a,b,s); }
    }
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        const __m512d t = _mm512_mask_load_pd( src, k, &(a[i]) );
        _mm512_mask_store_pd( &(b[i]), k, t);
    }
}

static inline void stride2_mvmovupd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m512d src = {0};
    const __mmask8 k0 = 85; // 10101010
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=16) {
        const __m512d t0 = _mm512_mask_loadu_pd( src, k0, &(a[i  ]) );
        const __m512d t1 = _mm512_mask_loadu_pd( src, k0, &(a[i+8]) );
        _mm512_mask_storeu_pd( &(b[i  ]), k0, t0);
        _mm512_mask_storeu_pd( &(b[i+8]), k0, t1);
    }
}

static inline void stride3_mvmovupd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m512d src = {0};
    const __mmask8 k0 =  73; // 10010010
    const __mmask8 k1 = 146; // 01001001
    const __mmask8 k2 =  36; // 00100100
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=24) {
        const __m512d t0 = _mm512_mask_loadu_pd( src, k0, &(a[i   ]) );
        const __m512d t1 = _mm512_mask_loadu_pd( src, k1, &(a[i+ 8]) );
        const __m512d t2 = _mm512_mask_loadu_pd( src, k2, &(a[i+16]) );
        _mm512_mask_storeu_pd( &(b[i   ]), k0, t0);
        _mm512_mask_storeu_pd( &(b[i+ 8]), k1, t1);
        _mm512_mask_storeu_pd( &(b[i+16]), k2, t2);
    }
}

static inline void stride5_mvmovupd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m512d src = {0};
    const __mmask8 k0 =  33; // 10000100
    const __mmask8 k1 = 132; // 00100001
    const __mmask8 k2 =  16; // 00001000
    const __mmask8 k3 =  66; // 01000010
    const __mmask8 k4 =   8; // 00001000
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=40) {
        const __m512d t0 = _mm512_mask_loadu_pd( src, k0, &(a[i   ]) );
        const __m512d t1 = _mm512_mask_loadu_pd( src, k1, &(a[i+ 8]) );
        const __m512d t2 = _mm512_mask_loadu_pd( src, k2, &(a[i+16]) );
        const __m512d t3 = _mm512_mask_loadu_pd( src, k3, &(a[i+24]) );
        const __m512d t4 = _mm512_mask_loadu_pd( src, k4, &(a[i+32]) );
        _mm512_mask_storeu_pd( &(b[i   ]), k0, t0);
        _mm512_mask_storeu_pd( &(b[i+ 8]), k1, t1);
        _mm512_mask_storeu_pd( &(b[i+16]), k2, t2);
        _mm512_mask_storeu_pd( &(b[i+24]), k3, t3);
        _mm512_mask_storeu_pd( &(b[i+32]), k4, t4);
    }
}

void stride_mvmovupd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    const __m512d src = {0};
    __mmask8 k = 255;
    switch (s) {
        case 1: k = 255; break;
        //case 2: k =  85; break;
        case 2: { return stride2_mvmovupd512(n,a,b); }
        case 3: { return stride3_mvmovupd512(n,a,b); }
        case 4: k =  17; break;
        case 5: { return stride5_mvmovupd512(n,a,b); }
        case 8: k =   1; break;
        default: { return stride_ref(n,a,b,s); }
    }
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        const __m512d t = _mm512_mask_loadu_pd( src, k, &(a[i]) );
        _mm512_mask_storeu_pd( &(b[i]), k, t);
    }
}

void stride_vGSdpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    const __m256i vindex = _mm256_set_epi32(7*s,6*s,5*s,4*s,3*s,2*s,s,0); // start from the right...
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=(8*s)) {
        const __m512d t = _mm512_i32gather_pd(vindex, &(a[i]), 8 /* scale */ );
        _mm512_i32scatter_pd( &(b[i]), vindex, t, 8 /* scale */ );
    }
}

void stride_vGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
    const __m512i vindex = _mm512_set_epi64(7*s,6*s,5*s,4*s,3*s,2*s,s,0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=(8*s)) {
        const __m512d t = _mm512_i64gather_pd(vindex, &(a[i]), 8 /* scale */ );
        _mm512_i64scatter_pd( &(b[i]), vindex, t, 8 /* scale */ );
    }
}

void stride_mvGSdpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
    const __m512d src = {0};
    const __mmask8 k =  255;
    const __m256i vindex = _mm256_set_epi32(7*s,6*s,5*s,4*s,3*s,2*s,s,0);
    const int hint = _MM_HINT_NONE;
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=(8*s)) {
        const __m512d t = _mm512_mask_i32gather_pd(src, k, vindex, &(a[i]), 8 /* scale */ );
        _mm512_mask_i32scatter_pd( &(b[i]), k, vindex, t, 8 /* scale */ );
    }
}

void stride_mvGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
    const __m512d src = {0};
    const __mmask8 k =  255;
    const __m512i vindex = _mm512_set_epi64(7*s,6*s,5*s,4*s,3*s,2*s,s,0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=(8*s)) {
        const __m512d t = _mm512_mask_i64gather_pd(src, k, vindex, &(a[i]), 8 /* scale */ );
        _mm512_mask_i64scatter_pd( &(b[i]), k, vindex, t, 8 /* scale */ );
    }
}
#endif /* AVX-512F */

#ifdef __AVX512PF__
void stride_vPFGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s)
{
    ASSUME(s>0);
    const __m512i vindex = _mm512_set_epi64(7*s,6*s,5*s,4*s,3*s,2*s,s,0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=(8*s)) {
        const __m512d t = _mm512_i64gather_pd(vindex, &(a[i]), 8 /* scale */ );
        _mm512_i64scatter_pd( &(b[i]), vindex, t, 8 /* scale */ );
        _mm512_prefetch_i64gather_pd(vindex, &(a[i+8*s]), 8 /* scale */, _MM_HINT_T0);
        _mm512_prefetch_i64scatter_pd( &(b[i+8*s]), vindex, 8 /* scale */, _MM_HINT_T0);
    }
}
#endif /* AVX-512PF */
