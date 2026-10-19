#ifndef NAN_COMMON_H
#define NAN_COMMON_H

/*
 * Shared declarations for the NaN add/mul/min/max tests.
 *
 * Every backend applies each operation to the two operand orders
 *     (1, NaN)  and  (NaN, 1)
 * and reports both results through nt_record().  SIMD backends put the
 * first order in lane 0 and the second in lane 1 of one vector operation,
 * so a single instruction tests both orders.
 */

#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* enough lanes for the widest SVE (2048 bits = 32 doubles) */
#define NT_N 32

/* nt_a = {1, NaN, 1, NaN, ...}   nt_b = {NaN, 1, NaN, 1, ...}
 * filled at run time so the compiler cannot constant-fold the tests */
extern double nt_a[NT_N];
extern double nt_b[NT_N];

/* r_x_nan = op(1, NaN),  r_nan_x = op(NaN, 1).  op is "add","mul","max","min" */
void nt_record(const char *impl, const char *op, double r_x_nan, double r_nan_x);
void nt_skip(const char *impl, const char *why);
void nt_note(const char *text);

/* ---- which backends exist in this build ---- */
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#define NT_X86 1
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#define NT_NEON 1
#endif
/* NT_SVE, NT_CUDA, NT_OPENCL, HAVE_FMAXIMUM are set by the Makefile */

#ifdef NT_X86
void nt_run_x86(void);
#endif
#ifdef NT_NEON
void nt_run_neon(void);
#endif
#if defined(__aarch64__)
int nt_has_sve(void); /* run-time check, compiled without SVE flags */
#endif
#ifdef NT_SVE
void nt_run_sve(void);
#endif
#ifdef NT_CUDA
void nt_run_cuda(void);
#endif
#ifdef NT_OPENCL
void nt_run_opencl(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
