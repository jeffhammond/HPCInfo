#include "triad.h"

void triad_ref(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
OMP_PARALLEL_FOR
    for (size_t i=0; i<n; i++) {
        c[i] = s * a[i] + b[i];
    }
}

#ifdef __SSE2__

void triad_movapd128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        //_mm_empty();
        __m128d ts = _mm_load1_pd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=2) {
            __m128d ta = _mm_load_pd(&(a[i]));
            __m128d tb = _mm_load_pd(&(b[i]));
                    ta = _mm_mul_pd(ts,ta);
            __m128d tc = _mm_add_pd(ta,tb);
            _mm_store_pd(&(c[i]), tc);
        }
    }
}

void triad_movntpd128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        //_mm_empty();
        __m128d ts = _mm_load1_pd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=2) {
            __m128d ta = _mm_load_pd(&(a[i]));
            __m128d tb = _mm_load_pd(&(b[i]));
                    ta = _mm_mul_pd(ts,ta);
            __m128d tc = _mm_add_pd(ta,tb);
            _mm_stream_pd(&(c[i]), tc);
        }
        _mm_sfence();
    }
}

#endif /* SSE2 */

#ifdef __SSE4_1__
void triad_movntdq128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        //_mm_empty();
        __m128d ts = _mm_load1_pd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=2) {
            __m128d ta = _mm_load_pd(&(a[i]));
            __m128d tb = _mm_load_pd(&(b[i]));
                    ta = _mm_mul_pd(ts,ta);
            __m128d tc = _mm_add_pd(ta,tb);
            _mm_stream_si128((__m128i*)&(c[i]),(__m128i)tc);
        }
        _mm_sfence();
    }
}
#endif /* SSE4.1 */

#ifdef __AVX__
void triad_vmovapd256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m256d ts = _mm256_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=4) {
            __m256d ta = _mm256_load_pd(&(a[i]));
            __m256d tb = _mm256_load_pd(&(b[i]));
                    ta = _mm256_mul_pd(ts,ta);
            __m256d tc = _mm256_add_pd(ta,tb);
            _mm256_store_pd(&(c[i]), tc);
        }
    }
}

void triad_vmovntpd256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m256d ts = _mm256_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=4) {
            __m256d ta = _mm256_load_pd(&(a[i]));
            __m256d tb = _mm256_load_pd(&(b[i]));
                    ta = _mm256_mul_pd(ts,ta);
            __m256d tc = _mm256_add_pd(ta,tb);
            _mm256_stream_pd(&(c[i]), tc);
        }
        _mm_sfence();
    }
}
#endif /* AVX */

#ifdef __AVX2__
void triad_vmovntdqa256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m256d ts = _mm256_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=4) {
            __m256d ta = (__m256d) _mm256_stream_load_si256((__m256i*)&(a[i]));
            __m256d tb = (__m256d) _mm256_stream_load_si256((__m256i*)&(b[i]));
                    ta = _mm256_mul_pd(ts,ta);
            __m256d tc = _mm256_add_pd(ta,tb);
            _mm256_stream_pd(&(c[i]), tc);
        }
        _mm_sfence();
    }
}
#endif /* AVX2 */

#ifdef __AVX512F__
void triad_vmovapd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m512d ts = _mm512_broadcast_sd(&s);
        for (size_t i=0; i<n; i+=8) {
            __m512d ta = _mm512_load_pd( &(a[i]) );
            __m512d tb = _mm512_load_pd( &(b[i]) );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_store_pd( &(c[i]), tc);
        }
    }
}

void triad_vmovupd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m512d ts = _mm512_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=8) {
            __m512d ta = _mm512_loadu_pd( &(a[i]) );
            __m512d tb = _mm512_loadu_pd( &(b[i]) );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_storeu_pd( &(c[i]), tc);
        }
    }
}

void triad_mvmovapd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m512d src = {0};
        __mmask8 k = 255;
        __m512d ts = _mm512_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=8) {
            __m512d ta = _mm512_mask_load_pd( src, k, &(ta[i]) );
            __m512d tb = _mm512_mask_load_pd( src, k, &(tb[i]) );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_mask_store_pd( &(c[i]), k, tc);
        }
    }
}

void triad_mvmovupd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m512d src = {0};
        __mmask8 k = 255;
        __m512d ts = _mm512_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=8) {
            __m512d ta = _mm512_mask_loadu_pd( src, k, &(ta[i]) );
            __m512d tb = _mm512_mask_loadu_pd( src, k, &(tb[i]) );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_mask_storeu_pd( &(c[i]), k, tc);
        }
    }
}

void triad_vmovntpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m512d ts = _mm512_broadcast_sd(&s);
        for (size_t i=0; i<n; i+=8) {
            __m512d ta = _mm512_load_pd( &(a[i]) );
            __m512d tb = _mm512_load_pd( &(b[i]) );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_stream_pd( &(c[i]), tc);
        }
        _mm_sfence();
    }
}

void triad_vmovntdqa512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m512d ts = _mm512_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=8) {
            __m512i ta = _mm512_stream_load_si512( (__m512i*)&(a[i]) );
            __m512i tb = _mm512_stream_load_si512( (__m512i*)&(b[i]) );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_stream_si512 ( (__m512i*)&(c[i]), tc);
        }
        _mm_sfence();
    }
}

void triad_vGSdpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        const __m512i vindex = _mm512_set_epi32(7,6,5,4,3,2,1,0); // start from the right...
        __m512d ts = _mm512_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=8) {
            __m512d ta = _mm512_i32gather_pd(vindex, &(a[i]), 8 /* scale */ );
            __m512d tb = _mm512_i32gather_pd(vindex, &(b[i]), 8 /* scale */ );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_i32scatter_pd( &(c[i]), vindex, tc, 8 /* scale */ );
        }
    }

void triad_mvGSdpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m512d src = {0};
        __mmask8 k = 255;
        const __m512i vindex = _mm512_set_epi32(7,6,5,4,3,2,1,0); // start from the right...
        __m512d ts = _mm512_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=8) {
            __m512d ta = _mm512_mask_i32gather_pd(src, k, vindex, &(a[i]), 8 /* scale */ );
            __m512d tb = _mm512_mask_i32gather_pd(src, k, vindex, &(b[i]), 8 /* scale */ );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_mask_i32scatter_pd( &(c[i]), k, vindex, tc, 8 /* scale */ );
        }
    }
}

void triad_vGSqpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        const __m512i vindex = _mm512_set_epi64(7,6,5,4,3,2,1,0);
        __m512d ts = _mm512_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=8) {
            __m512d ta = _mm512_i64gather_pd(vindex, &(a[i]), 8 /* scale */ );
            __m512d tb = _mm512_i64gather_pd(vindex, &(b[i]), 8 /* scale */ );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_i64scatter_pd( &(c[i]), vindex, tc, 8 /* scale */ );
        }
    }
}

void triad_mvGSqpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c)
{
    OMP_PARALLEL
    {
        __m512d src = {0};
        __mmask8 k = 255;
        const __m512i vindex = _mm512_set_epi64(7,6,5,4,3,2,1,0);
        __m512d ts = _mm512_broadcast_sd(&s);
        OMP_FOR
        for (size_t i=0; i<n; i+=8) {
            __m512d ta = _mm512_mask_i64gather_pd(src, k, vindex, &(a[i]), 8 /* scale */ );
            __m512d tb = _mm512_mask_i64gather_pd(src, k, vindex, &(b[i]), 8 /* scale */ );
                    ta = _mm512_mul_pd(ts,ta);
            __m512d tc = _mm512_add_pd(ta,tb);
            _mm512_mask_i64scatter_pd( &(c[i]), k, vindex, tc, 8 /* scale */ );
        }
    }
}
#endif /* AVX-512F */
