#include "copy.h"

void copy_ref(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i++) {
        b[i] = a[i];
    }
}

void copy_mov(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i++) {
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

#if __aarch64__
void copy_vld1q(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=2) {
        float64x2_t t = vld1q_f64(&a[i]);
        vst1q_f64(&b[i], t);
    }
}
#endif

#if __x86_64__
void copy_rep_movsq(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    /* It might make more sense to do rep-movsq a page at a time
     * and make the alignment nicer... */
#ifdef _OPENMP
#pragma omp parallel
    {
        int me = omp_get_thread_num();
        int nt = omp_get_num_threads();
        size_t chunk = 1+(n-1)/nt;
        size_t start = me*chunk;
        size_t end   = (me+1)*chunk;
        if (end>n) end = n;
        size_t tn =  (end>start) ? end-start : 0;
        //const double * RESTRICT ta = &( a[start] );
        //      double * RESTRICT tb = &( b[start] );
        const double * RESTRICT ta = a+start;
              double * RESTRICT tb = b+start;
        //printf("zzz %d: chunk=%zu\n", me, chunk); fflush(stdout);
        //printf("zzz %d: start=%zu\n", me, start); fflush(stdout);
        //printf("zzz %d: xend=%zu\n", me, end); fflush(stdout);
        //printf("zzz %d: count=%zd\n", me, tn); fflush(stdout);
#ifdef __INTEL_COMPILER
        asm("rep movsq"
            : "=D" (tb), "=S" (ta), "=c" (tn)
            : "0" (tb), "1" (ta), "2" (tn)
            : "memory");
#else
        tn *= sizeof(double);
        memcpy(tb,ta,tn);
#endif
    }
#else
    {
#if HAS_GNU_EXTENDED_ASM
        asm("rep movsq"
            : "=D" (b), "=S" (a), "=c" (n)
            : "0" (b), "1" (a), "2" (n)
            : "memory");
#else
        tn *= sizeof(double);
        memcpy(b,a,n*sizeof(double));
#endif
    }
#endif
}
#endif // __x86_64__

#ifdef __SSE__

#if 0 /* BROKEN */
void copy_movntq(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i++) {
        double t;
        //t = a[i];
        asm ("mov %1, %0" : "=r" (t) : "m" (a[i]));
        //b[i] = t;
        // movntq does not work here...
        asm ("movntq %1, %0" : "=m" (b[i]) : "r" (t));
    }
    asm ("sfence" ::: "memory");
}
#endif

#ifdef __INTEL_COMPILER
void copy_movntq64(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    //_mm_empty();
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i++) {
        __m64 t = _m_from_int64( *(__int64*)&(a[i]) );
        _mm_stream_pi( (__m64*)&(b[i]), (__m64)t);
    }
    _mm_sfence();
}
#endif /* ICC */

#endif /* SSE */

#ifdef __SSE2__
void copy_movnti(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i++) {
        double t;
        //t = a[i];
        asm ("mov %1, %0" : "=r" (t) : "m" (a[i]));
        //b[i] = t;
        asm ("movnti %1, %0" : "=m" (b[i]) : "r" (t));
    }
    asm ("sfence" ::: "memory");
}

#ifdef __INTEL_COMPILER
void copy_movnti64(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    //_mm_empty();
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i++) {
        __m64 t = _m_from_int64( *(__int64*)&(a[i]) );
        _mm_stream_si64( (__int64*)&(b[i]), *(__int64*)&t);
    }
    _mm_sfence();
}
#endif /* ICC */

void copy_movapd128(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=2) {
        __m128d t = _mm_load_pd( &(a[i]) );
        _mm_store_pd( &(b[i]), t);
    }
}

void copy_movntpd128(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=2) {
        __m128d t = _mm_load_pd( &(a[i]) );
        _mm_stream_pd( &(b[i]), t);
    }
    _mm_sfence();
}
#endif /* SSE2 */

#ifdef __SSE4_1__
void copy_movntdqa128(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=2) {
        __m128i t = _mm_stream_load_si128( (__m128i*)&(a[i]) );
        _mm_stream_si128 ( (__m128i*)&(b[i]), t);
    }
    _mm_sfence();
}
#endif /* SSE4.1 */

#ifdef __AVX__
void copy_vmovapd256(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=4) {
        __m256d t = _mm256_load_pd( &(a[i]) );
        _mm256_store_pd( &(b[i]), t);
    }
}

void copy_vmovntpd256(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=4) {
        __m256d t = _mm256_load_pd( &(a[i]) );
        _mm256_stream_pd( &(b[i]), t);
    }
    _mm_sfence();
}
#endif /* AVX */

#ifdef __AVX2__
void copy_vmovntdqa256(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=4) {
        __m256i t = _mm256_stream_load_si256( (__m256i*)&(a[i]) );
        _mm256_stream_si256 ( (__m256i*)&(b[i]), t);
    }
    _mm_sfence();
}

void copy_vgatherdpd128(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m128i vindex = _mm_set_epi32(-1,-1,1,0); // start from the right...
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=2) {
        __m128d t = _mm_i32gather_pd( &(a[i]), vindex, 8 /* scale */ );
        _mm_storel_pd( &(b[i  ]), t);
        _mm_storeh_pd( &(b[i+1]), t);
    }
}

void copy_vgatherqpd128(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m128i vindex = _mm_set_epi64x(1,0); // works
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=2) {
        __m128d t = _mm_i64gather_pd( &(a[i]), vindex, 8 /* scale */ );
        _mm_storel_pd( &(b[i  ]), t);
        _mm_storeh_pd( &(b[i+1]), t);
    }
}

void copy_vgatherdpd256(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m128i vindex = _mm_set_epi32(3,2,1,0); // start from the right...
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=4) {
        __m256d t = _mm256_i32gather_pd( &(a[i]), vindex, 8 /* scale */ );
        __m128d l = _mm256_extractf128_pd(t,0);
        __m128d u = _mm256_extractf128_pd(t,1);
        _mm_storel_pd( &(b[i  ]), l);
        _mm_storeh_pd( &(b[i+1]), l);
        _mm_storel_pd( &(b[i+2]), u);
        _mm_storeh_pd( &(b[i+3]), u);
    }
}

void copy_vgatherqpd256(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m256i vindex = _mm256_set_epi64x(3,2,1,0); // works
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=4) {
        __m256d t = _mm256_i64gather_pd( &(a[i]), vindex, 8 /* scale */ );
        __m128d l = _mm256_extractf128_pd(t,0);
        __m128d u = _mm256_extractf128_pd(t,1);
        _mm_storel_pd( &(b[i  ]), l);
        _mm_storeh_pd( &(b[i+1]), l);
        _mm_storel_pd( &(b[i+2]), u);
        _mm_storeh_pd( &(b[i+3]), u);
    }
}

void copy_mvgatherqpd256(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m256i vindex = _mm256_set_epi64x(3,2,1,0); // works
    // O in OQ means ordered, i.e. AND.  unordered is OR.  Q means quiet i.e. non-signaling.
    __m256d src = _mm256_cmp_pd(_mm256_setzero_pd(),_mm256_setzero_pd(),_CMP_EQ_OQ); // sets all bits to 1
    __m256d mask = src;
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=4) {
        __m256d t = _mm256_mask_i64gather_pd( src, &(a[i]), vindex, mask, 8 /* scale */ );
        __m128d l = _mm256_extractf128_pd(t,0);
        __m128d u = _mm256_extractf128_pd(t,1);
        _mm_storel_pd( &(b[i  ]), l);
        _mm_storeh_pd( &(b[i+1]), l);
        _mm_storel_pd( &(b[i+2]), u);
        _mm_storeh_pd( &(b[i+3]), u);
    }
}
#endif /* AVX2 */

#ifdef __AVX512F__
void copy_vmovapd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512d t = _mm512_load_pd( &(a[i]) );
        _mm512_store_pd( &(b[i]), t);
    }
}

void copy_vmovupd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512d t = _mm512_loadu_pd( &(a[i]) );
        _mm512_storeu_pd( &(b[i]), t);
    }
}

void copy_mvmovapd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    __m512d src = {0};
    __mmask8 k = 255;
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512d t = _mm512_mask_load_pd( src, k, &(a[i]) );
        _mm512_mask_store_pd( &(b[i]), k, t);
    }
}

void copy_mvmovupd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    __m512d src = {0};
    __mmask8 k = 255;
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512d t = _mm512_mask_loadu_pd( src, k, &(a[i]) );
        _mm512_mask_storeu_pd( &(b[i]), k, t);
    }
}

void copy_vmovntpd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512d t = _mm512_load_pd( &(a[i]) );
        _mm512_stream_pd( &(b[i]), t);
    }
    _mm_sfence();
}

void copy_vmovntdqa512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512i t = _mm512_stream_load_si512( (__m512i*)&(a[i]) );
        _mm512_stream_si512 ( (__m512i*)&(b[i]), t);
    }
    _mm_sfence();
}

void copy_vGSdpd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m256i vindex = _mm256_set_epi32(7,6,5,4,3,2,1,0); // start from the right...
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512d t = _mm512_i32gather_pd(vindex, &(a[i]), 8 /* scale */ );
        _mm512_i32scatter_pd( &(b[i]), vindex, t, 8 /* scale */ );
    }
}

void copy_mvGSdpd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    __m512d src = {0};
    __mmask8 k = 255;
    const __m256i vindex = _mm256_set_epi32(7,6,5,4,3,2,1,0); // start from the right...
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512d t = _mm512_mask_i32gather_pd(src, k, vindex, &(a[i]), 8 /* scale */ );
        _mm512_mask_i32scatter_pd( &(b[i]), k, vindex, t, 8 /* scale */ );
    }
}

void copy_vGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    const __m512i vindex = _mm512_set_epi64(7,6,5,4,3,2,1,0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512d t = _mm512_i64gather_pd(vindex, &(a[i]), 8 /* scale */ );
        _mm512_i64scatter_pd( &(b[i]), vindex, t, 8 /* scale */ );
    }
}

void copy_mvGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b)
{
    __m512d src = {0};
    __mmask8 k = 255;
    const __m512i vindex = _mm512_set_epi64(7,6,5,4,3,2,1,0);
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i+=8) {
        __m512d t = _mm512_mask_i64gather_pd(src, k, vindex, &(a[i]), 8 /* scale */ );
        _mm512_mask_i64scatter_pd( &(b[i]), k, vindex, t, 8 /* scale */ );
    }
}
#endif /* AVX-512F */
