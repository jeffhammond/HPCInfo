#include <stdio.h>
#include <string.h>
#include <math.h>

#include "nan_common.h"

double nt_a[NT_N];
double nt_b[NT_N];

static int nt_bad;

static void fmtv(char *buf, size_t n, double v)
{
    if (isnan(v)) snprintf(buf, n, "NaN");
    else          snprintf(buf, n, "%g", v);
}

void nt_record(const char *impl, const char *op, double r1, double r2)
{
    const int n1 = isnan(r1), n2 = isnan(r2);
    const char *what;

    if (n1 && n2)        what = "propagates NaN";
    else if (!n1 && !n2) what = "ignores NaN";
    else if (n1)         what = "returns 2nd operand (asymmetric)";
    else                 what = "returns 1st operand (asymmetric)";

    /* + and * must always give NaN; anything else is a real problem */
    if ((!strcmp(op, "add") || !strcmp(op, "mul")) && !(n1 && n2)) {
        what = "UNEXPECTED: NaN lost";
        nt_bad++;
    }

    char s1[32], s2[32];
    fmtv(s1, sizeof s1, r1);
    fmtv(s2, sizeof s2, r2);
    printf("  %-3s %-34s (1,NaN)->%-4s (NaN,1)->%-4s %s\n", op, impl, s1, s2, what);
}

void nt_skip(const char *impl, const char *why)
{
    printf("  --- %-34s skipped: %s\n", impl, why);
}

void nt_note(const char *text)
{
    printf("  [%s]\n", text);
}

/* ---- portable scalar C ---- */

typedef double (*binop)(double, double);

static double op_add(double a, double b)  { return a + b; }
static double op_mul(double a, double b)  { return a * b; }
static double op_fmax(double a, double b) { return fmax(a, b); }
static double op_fmin(double a, double b) { return fmin(a, b); }
static double op_tmax(double a, double b) { return (a > b) ? a : b; }
static double op_tmin(double a, double b) { return (a < b) ? a : b; }
#ifdef HAVE_FMAXIMUM
static double op_fmaximum(double a, double b)     { return fmaximum(a, b); }
static double op_fminimum(double a, double b)     { return fminimum(a, b); }
static double op_fmaximum_num(double a, double b) { return fmaximum_num(a, b); }
static double op_fminimum_num(double a, double b) { return fminimum_num(a, b); }
#endif

static const struct {
    const char *impl, *op;
    binop f;
} scalar_ops[] = {
    {"C a + b",             "add", op_add},
    {"C a * b",             "mul", op_mul},
    {"C fmax()",            "max", op_fmax},
    {"C fmin()",            "min", op_fmin},
    {"C a > b ? a : b",     "max", op_tmax},
    {"C a < b ? a : b",     "min", op_tmin},
#ifdef HAVE_FMAXIMUM
    {"C23 fmaximum()",      "max", op_fmaximum},
    {"C23 fminimum()",      "min", op_fminimum},
    {"C23 fmaximum_num()",  "max", op_fmaximum_num},
    {"C23 fminimum_num()",  "min", op_fminimum_num},
#endif
};

int main(void)
{
    /* runtime values: keep the compiler from folding anything */
    volatile double one = 1.0;
    volatile double nan = NAN;
    for (int i = 0; i < NT_N; i++) {
        nt_a[i] = (i % 2 == 0) ? one : nan;
        nt_b[i] = (i % 2 == 0) ? nan : one;
    }

    printf("Each row applies the op to (1,NaN) and (NaN,1).\n");
    printf("  add/mul must propagate NaN; for max/min the interesting question is what happens.\n");

    printf("\n== portable C ==\n");
    for (size_t i = 0; i < sizeof scalar_ops / sizeof scalar_ops[0]; i++)
        nt_record(scalar_ops[i].impl, scalar_ops[i].op,
                  scalar_ops[i].f(nt_a[0], nt_b[0]),
                  scalar_ops[i].f(nt_a[1], nt_b[1]));

#ifdef NT_X86
    printf("\n== x86: x87 / SSE2 / AVX / AVX-512 ==\n");
    nt_run_x86();
#endif
#ifdef NT_NEON
    printf("\n== Arm: NEON ==\n");
    nt_run_neon();
#endif
#ifdef NT_SVE
    printf("\n== Arm: SVE ==\n");
    if (nt_has_sve()) nt_run_sve();
    else              nt_skip("SVE", "CPU/OS does not report SVE");
#endif
#ifdef NT_CUDA
    printf("\n== CUDA ==\n");
    nt_run_cuda();
#endif
#ifdef NT_OPENCL
    printf("\n== OpenCL ==\n");
    nt_run_opencl();
#endif

    if (nt_bad) printf("\n%d UNEXPECTED result(s): add/mul lost a NaN\n", nt_bad);
    return nt_bad != 0;
}
