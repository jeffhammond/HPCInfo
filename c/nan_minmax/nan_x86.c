#include "nan_common.h"

#ifdef NT_X86

#include <float.h>
#include <immintrin.h>

/* Each ISA level lives in a function carrying its own target attribute, so
 * one binary holds all of them and the CPU is queried before any is called. */
#define TGT(x) __attribute__((target(x)))

/* ---- x87 (long double is the 80-bit x87 type on x86 GCC/clang) ---- */

#if LDBL_MANT_DIG == 64
static long double ld_add(long double a, long double b) { return a + b; }
static long double ld_mul(long double a, long double b) { return a * b; }
static long double ld_tmax(long double a, long double b) { return (a > b) ? a : b; }
static long double ld_tmin(long double a, long double b) { return (a < b) ? a : b; }
static long double ld_fmax(long double a, long double b) { return fmaxl(a, b); }
static long double ld_fmin(long double a, long double b) { return fminl(a, b); }

static void run_x87(void)
{
    /* volatile 80-bit copies: otherwise the compiler sees they are widened
     * doubles and narrows the ternaries to SSE maxsd/minsd, which is not x87 */
    volatile long double va0 = nt_a[0], vb0 = nt_b[0], va1 = nt_a[1], vb1 = nt_b[1];
    const long double a0 = va0, b0 = vb0, a1 = va1, b1 = vb1;
#define X87(name, op, f) nt_record(name, op, (double)f(a0, b0), (double)f(a1, b1))
    X87("x87 fadd",              "add", ld_add);
    X87("x87 fmul",              "mul", ld_mul);
    X87("x87 a>b?a:b",           "max", ld_tmax);
    X87("x87 a<b?a:b",           "min", ld_tmin);
    X87("x87 fmaxl()",           "max", ld_fmax);
    X87("x87 fminl()",           "min", ld_fmin);
#undef X87
}
#endif

/* ---- SSE2: scalar (sd) and packed (pd) ---- */

static TGT("sse2") void run_sse2(void)
{
#define SD(name, op, intr) do {                                          \
        double r0, r1;                                                   \
        _mm_store_sd(&r0, intr(_mm_load_sd(&nt_a[0]), _mm_load_sd(&nt_b[0]))); \
        _mm_store_sd(&r1, intr(_mm_load_sd(&nt_a[1]), _mm_load_sd(&nt_b[1]))); \
        nt_record(name, op, r0, r1);                                     \
    } while (0)
    SD("SSE2 addsd", "add", _mm_add_sd);
    SD("SSE2 mulsd", "mul", _mm_mul_sd);
    SD("SSE2 maxsd", "max", _mm_max_sd);
    SD("SSE2 minsd", "min", _mm_min_sd);
#undef SD

#define PD(name, op, intr) do {                                          \
        double r[2];                                                     \
        _mm_storeu_pd(r, intr(_mm_loadu_pd(nt_a), _mm_loadu_pd(nt_b)));  \
        nt_record(name, op, r[0], r[1]);                                 \
    } while (0)
    PD("SSE2 addpd", "add", _mm_add_pd);
    PD("SSE2 mulpd", "mul", _mm_mul_pd);
    PD("SSE2 maxpd", "max", _mm_max_pd);
    PD("SSE2 minpd", "min", _mm_min_pd);
#undef PD
}

/* ---- AVX: 256-bit ---- */

static TGT("avx") void run_avx(void)
{
#define PD4(name, op, intr) do {                                               \
        double r[4];                                                           \
        _mm256_storeu_pd(r, intr(_mm256_loadu_pd(nt_a), _mm256_loadu_pd(nt_b))); \
        nt_record(name, op, r[0], r[1]);                                       \
    } while (0)
    PD4("AVX vaddpd ymm", "add", _mm256_add_pd);
    PD4("AVX vmulpd ymm", "mul", _mm256_mul_pd);
    PD4("AVX vmaxpd ymm", "max", _mm256_max_pd);
    PD4("AVX vminpd ymm", "min", _mm256_min_pd);
#undef PD4
}

/* ---- AVX-512 ---- */

#ifndef NT_NO_AVX512
static TGT("avx512f") void run_avx512f(void)
{
#define PD8(name, op, intr) do {                                               \
        double r[8];                                                           \
        _mm512_storeu_pd(r, intr(_mm512_loadu_pd(nt_a), _mm512_loadu_pd(nt_b))); \
        nt_record(name, op, r[0], r[1]);                                       \
    } while (0)
    PD8("AVX-512F vaddpd zmm", "add", _mm512_add_pd);
    PD8("AVX-512F vmulpd zmm", "mul", _mm512_mul_pd);
    PD8("AVX-512F vmaxpd zmm", "max", _mm512_max_pd);
    PD8("AVX-512F vminpd zmm", "min", _mm512_min_pd);
#undef PD8
}

/* VRANGEPD imm8: [1:0] = 00 min, 01 max;  [3:2] = 01 sign from the compare.
 * 0x04 / 0x05 are therefore the plain min / max. */
static TGT("avx512f,avx512dq") void run_avx512dq(void)
{
    double r[8];
    __m512d x = _mm512_loadu_pd(nt_a), y = _mm512_loadu_pd(nt_b);
    _mm512_storeu_pd(r, _mm512_range_pd(x, y, 0x05));
    nt_record("AVX-512DQ vrangepd max", "max", r[0], r[1]);
    _mm512_storeu_pd(r, _mm512_range_pd(x, y, 0x04));
    nt_record("AVX-512DQ vrangepd min", "min", r[0], r[1]);
}
#endif

void nt_run_x86(void)
{
    __builtin_cpu_init();

#if LDBL_MANT_DIG == 64
    run_x87();
#else
    nt_skip("x87", "long double is not the 80-bit x87 type here");
#endif

    if (__builtin_cpu_supports("sse2")) run_sse2();
    else nt_skip("SSE2", "CPU lacks SSE2");

    if (__builtin_cpu_supports("avx")) run_avx();
    else nt_skip("AVX", "CPU lacks AVX");

#ifndef NT_NO_AVX512
    if (__builtin_cpu_supports("avx512f")) run_avx512f();
    else nt_skip("AVX-512F", "CPU lacks AVX-512F");

    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")) run_avx512dq();
    else nt_skip("AVX-512DQ vrangepd", "CPU lacks AVX-512DQ");
#else
    nt_skip("AVX-512", "built with NT_NO_AVX512");
#endif
}

#endif /* NT_X86 */
