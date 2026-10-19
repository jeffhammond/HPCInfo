/* Compiled with -march=armv8-a+sve; only called after nt_has_sve() succeeds. */
#include "nan_common.h"

#ifdef NT_SVE

#include <arm_sve.h>

void nt_run_sve(void)
{
    char note[64];
    snprintf(note, sizeof note, "SVE vector length = %u bits", (unsigned)svcntd() * 64u);
    nt_note(note);

    const svbool_t pg = svptrue_b64();
    const svfloat64_t a = svld1_f64(pg, nt_a), b = svld1_f64(pg, nt_b);
    double r[NT_N];

    /* lane 0 = (1,NaN), lane 1 = (NaN,1), repeating across the vector */
#define V(name, op, expr) do { svst1_f64(pg, r, expr); nt_record(name, op, r[0], r[1]); } while (0)
    V("SVE fadd",    "add", svadd_f64_x(pg, a, b));
    V("SVE fmul",    "mul", svmul_f64_x(pg, a, b));
    V("SVE fmax",    "max", svmax_f64_x(pg, a, b));
    V("SVE fmin",    "min", svmin_f64_x(pg, a, b));
    V("SVE fmaxnm",  "max", svmaxnm_f64_x(pg, a, b));
    V("SVE fminnm",  "min", svminnm_f64_x(pg, a, b));
#undef V
}

#endif /* NT_SVE */
