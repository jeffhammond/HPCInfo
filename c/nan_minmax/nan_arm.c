#include "nan_common.h"

#ifdef NT_NEON

#include <arm_neon.h>

void nt_run_neon(void)
{
    const float64x2_t a = vld1q_f64(nt_a), b = vld1q_f64(nt_b);
    double r[2];

    /* 128-bit vector forms: lane 0 = (1,NaN), lane 1 = (NaN,1) */
#define V(name, op, expr) do { vst1q_f64(r, expr); nt_record(name, op, r[0], r[1]); } while (0)
    V("NEON fadd v.2d",    "add", vaddq_f64(a, b));
    V("NEON fmul v.2d",    "mul", vmulq_f64(a, b));
    V("NEON fmax v.2d",    "max", vmaxq_f64(a, b));
    V("NEON fmin v.2d",    "min", vminq_f64(a, b));
    V("NEON fmaxnm v.2d",  "max", vmaxnmq_f64(a, b));
    V("NEON fminnm v.2d",  "min", vminnmq_f64(a, b));
#undef V

    /* scalar d-register forms */
#define S(name, op, intr) do {                                       \
        float64x1_t x0 = vld1_f64(&nt_a[0]), y0 = vld1_f64(&nt_b[0]); \
        float64x1_t x1 = vld1_f64(&nt_a[1]), y1 = vld1_f64(&nt_b[1]); \
        nt_record(name, op, vget_lane_f64(intr(x0, y0), 0),          \
                            vget_lane_f64(intr(x1, y1), 0));         \
    } while (0)
    S("NEON fadd d",   "add", vadd_f64);
    S("NEON fmul d",   "mul", vmul_f64);
    S("NEON fmax d",   "max", vmax_f64);
    S("NEON fmin d",   "min", vmin_f64);
    S("NEON fmaxnm d", "max", vmaxnm_f64);
    S("NEON fminnm d", "min", vminnm_f64);
#undef S
}

#endif /* NT_NEON */

/* Kept here, not in nan_sve.c, because that file is compiled with SVE
 * enabled and nothing in it may run before this check passes. */
#if defined(__aarch64__)
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
int nt_has_sve(void)
{
#ifdef HWCAP_SVE
    return (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
#else
    return 0;
#endif
}
#else
int nt_has_sve(void) { return 0; }
#endif
#endif
