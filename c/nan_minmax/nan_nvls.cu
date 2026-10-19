// NVLink SHARP (NVLS) in-switch reductions via PTX multimem.ld_reduce and multimem.red,
// probing NaN behaviour.  Single process, all visible GPUs in one multicast object.
// Needs NVSwitch + fabric manager, a driver with multicast support, and sm_90+.
//
//   nvcc -O2 -arch=sm_90 nan_nvls.cu -o nan_nvls -lcuda && ./nan_nvls [-v]
//
// Every form that ptxas accepts for sm_90 is tested (see the *_TESTS lists).  The .NaN modifier,
// f32/f64 min/max, and integer ld_reduce add on s64 are rejected by ptxas, so they are absent.
//
// Each rank holds one operand per test slot; the cases are
//     finite        rank r holds r+1 (ints: see operand_bits)
//     NaN first     rank 0 holds a quiet NaN, the rest hold 1
//     NaN last      the last rank holds a quiet NaN, the rest hold 1
//     all NaN       every rank holds a quiet NaN
//     sNaN first    rank 0 holds a signaling NaN, the rest hold 1
//     NaN in mem    (red only) the target starts as a quiet NaN, every rank contributes 1
// ld_reduce: GPU 0 reduces across the ranks' copies and returns the value.
// red:       ranks 0..N-1 in turn reduce their operand into the shared location; we then read
//            every rank's replica of it.
// Default output is one line per test; -v prints every case.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CKR(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CKD(call) do { CUresult e_ = (call); if (e_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(e_, &s_); \
        fprintf(stderr, "driver error %s at %s:%d\n", s_, __FILE__, __LINE__); exit(1); } } while (0)

enum { OP_ADD, OP_MIN, OP_MAX, OP_AND, OP_OR, OP_XOR };
static const char *opname[] = { "add", "min", "max", "and", "or", "xor" };
enum { E_F32, E_F64, E_F16, E_BF16, E_U32, E_S32, E_U64, E_S64, E_B32, E_B64 };
enum { C_FINITE, C_NAN_FIRST, C_NAN_LAST, C_ALL_NAN, C_SNAN_FIRST, C_NAN_INIT, NC };
static const char *cname[NC] = {
    "finite", "NaN first", "NaN last", "all NaN", "sNaN first", "NaN in mem" };

#define SLOT     32          // bytes per (test, case) slot
#define RED_BASE 0x10000     // red targets live here; ld_reduce data lives at 0

// ---------------------------------------------------------------------------------------------
// Test lists.  X(id, op, element type, elements per slot, asm shape, "ptx suffix after the sem")
// The shape names the operand registers: R = .b32, F = .f32, D = .f64, L = .b64, H = .b16.
// ---------------------------------------------------------------------------------------------
#define PACKED_S(X, o, O, TY, E) /* f16x2 / bf16x2: scalar, v2, v4 */                          \
    X(o##_##TY##x2,    O, E, 2, R1, #o "." #TY "x2")                                           \
    X(o##_##TY##x2_v2, O, E, 4, R2, #o ".v2." #TY "x2")                                        \
    X(o##_##TY##x2_v4, O, E, 8, R4, #o ".v4." #TY "x2")
#define PACKED_V(X, o, O, TY, E) /* v2, v4 only */                                             \
    X(o##_##TY##x2_v2, O, E, 4, R2, #o ".v2." #TY "x2")                                        \
    X(o##_##TY##x2_v4, O, E, 8, R4, #o ".v4." #TY "x2")
#define HALF_V(X, o, O, TY, E)   /* f16 / bf16 vectors: v2, v4, v8 */                         \
    X(o##_##TY##_v2, O, E, 2, H2, #o ".v2." #TY)                                               \
    X(o##_##TY##_v4, O, E, 4, H4, #o ".v4." #TY)                                               \
    X(o##_##TY##_v8, O, E, 8, H8, #o ".v8." #TY)
#define ACC_PACKED(X, TY, E)                                                                   \
    X(add_acc_##TY##x2,    OP_ADD, E, 2, R1, "add.acc::f32." #TY "x2")                         \
    X(add_acc_##TY##x2_v2, OP_ADD, E, 4, R2, "add.acc::f32.v2." #TY "x2")                      \
    X(add_acc_##TY##x2_v4, OP_ADD, E, 8, R4, "add.acc::f32.v4." #TY "x2")
#define ACC_HALF(X, TY, E)                                                                     \
    X(add_acc_##TY##_v2, OP_ADD, E, 2, H2, "add.acc::f32.v2." #TY)                             \
    X(add_acc_##TY##_v4, OP_ADD, E, 4, H4, "add.acc::f32.v4." #TY)                             \
    X(add_acc_##TY##_v8, OP_ADD, E, 8, H8, "add.acc::f32.v8." #TY)
#define FLOAT_ADD(X)                                                                           \
    X(add_f32,    OP_ADD, E_F32, 1, F1, "add.f32")                                             \
    X(add_f32_v2, OP_ADD, E_F32, 2, F2, "add.v2.f32")                                          \
    X(add_f32_v4, OP_ADD, E_F32, 4, F4, "add.v4.f32")                                          \
    X(add_f64,    OP_ADD, E_F64, 1, D1, "add.f64")
#define INT_COMMON(X)                                                                          \
    X(add_u32, OP_ADD, E_U32, 1, R1, "add.u32")  X(add_s32, OP_ADD, E_S32, 1, R1, "add.s32")  \
    X(add_u64, OP_ADD, E_U64, 1, L1, "add.u64")                                                \
    X(min_u32, OP_MIN, E_U32, 1, R1, "min.u32")  X(min_s32, OP_MIN, E_S32, 1, R1, "min.s32")  \
    X(min_u64, OP_MIN, E_U64, 1, L1, "min.u64")  X(min_s64, OP_MIN, E_S64, 1, L1, "min.s64")  \
    X(max_u32, OP_MAX, E_U32, 1, R1, "max.u32")  X(max_s32, OP_MAX, E_S32, 1, R1, "max.s32")  \
    X(max_u64, OP_MAX, E_U64, 1, L1, "max.u64")  X(max_s64, OP_MAX, E_S64, 1, L1, "max.s64")
#define BITWISE(X)                                                                             \
    X(and_b32, OP_AND, E_B32, 1, R1, "and.b32")  X(and_b64, OP_AND, E_B64, 1, L1, "and.b64")  \
    X(or_b32,  OP_OR,  E_B32, 1, R1, "or.b32")   X(or_b64,  OP_OR,  E_B64, 1, L1, "or.b64")   \
    X(xor_b32, OP_XOR, E_B32, 1, R1, "xor.b32")  X(xor_b64, OP_XOR, E_B64, 1, L1, "xor.b64")

#define LDR_TESTS(X)                                                                           \
    FLOAT_ADD(X)                                                                               \
    PACKED_S(X, add, OP_ADD, f16, E_F16)  PACKED_S(X, add, OP_ADD, bf16, E_BF16)               \
    HALF_V(X, add, OP_ADD, f16, E_F16)    HALF_V(X, add, OP_ADD, bf16, E_BF16)                 \
    ACC_PACKED(X, f16, E_F16)  ACC_PACKED(X, bf16, E_BF16)                                     \
    ACC_HALF(X, f16, E_F16)    ACC_HALF(X, bf16, E_BF16)                                       \
    PACKED_S(X, min, OP_MIN, f16, E_F16)  PACKED_S(X, min, OP_MIN, bf16, E_BF16)               \
    HALF_V(X, min, OP_MIN, f16, E_F16)    HALF_V(X, min, OP_MIN, bf16, E_BF16)                 \
    PACKED_S(X, max, OP_MAX, f16, E_F16)  PACKED_S(X, max, OP_MAX, bf16, E_BF16)               \
    HALF_V(X, max, OP_MAX, f16, E_F16)    HALF_V(X, max, OP_MAX, bf16, E_BF16)                 \
    INT_COMMON(X) BITWISE(X)

// red has no scalar f16x2/bf16x2 min/max, and no scalar-vector ld_reduce add.s64
#define RED_TESTS(X)                                                                           \
    FLOAT_ADD(X)                                                                               \
    PACKED_S(X, add, OP_ADD, f16, E_F16)  PACKED_S(X, add, OP_ADD, bf16, E_BF16)               \
    HALF_V(X, add, OP_ADD, f16, E_F16)    HALF_V(X, add, OP_ADD, bf16, E_BF16)                 \
    ACC_PACKED(X, f16, E_F16)  ACC_PACKED(X, bf16, E_BF16)                                     \
    ACC_HALF(X, f16, E_F16)    ACC_HALF(X, bf16, E_BF16)                                       \
    PACKED_V(X, min, OP_MIN, f16, E_F16)  PACKED_V(X, min, OP_MIN, bf16, E_BF16)               \
    HALF_V(X, min, OP_MIN, f16, E_F16)    HALF_V(X, min, OP_MIN, bf16, E_BF16)                 \
    PACKED_V(X, max, OP_MAX, f16, E_F16)  PACKED_V(X, max, OP_MAX, bf16, E_BF16)               \
    HALF_V(X, max, OP_MAX, f16, E_F16)    HALF_V(X, max, OP_MAX, bf16, E_BF16)                 \
    INT_COMMON(X) BITWISE(X)

// ---------------------------------------------------------------------------------------------
// Device code
// ---------------------------------------------------------------------------------------------
#define LDRP "multimem.ld_reduce.relaxed.sys.global."
#define REDP "multimem.red.release.sys.global."

#define DEFLDR_R1(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    asm volatile(LDRP S " %0, [%1];" : "=r"(w[0]) : "l"(p) : "memory"); }
#define DEFLDR_R2(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    asm volatile(LDRP S " {%0,%1}, [%2];" : "=r"(w[0]), "=r"(w[1]) : "l"(p) : "memory"); }
#define DEFLDR_R4(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    asm volatile(LDRP S " {%0,%1,%2,%3}, [%4];" \
        : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "l"(p) : "memory"); }
#define DEFLDR_F1(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    float a; asm volatile(LDRP S " %0, [%1];" : "=f"(a) : "l"(p) : "memory"); \
    w[0] = __float_as_uint(a); }
#define DEFLDR_F2(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    float a, b; asm volatile(LDRP S " {%0,%1}, [%2];" : "=f"(a), "=f"(b) : "l"(p) : "memory"); \
    w[0] = __float_as_uint(a); w[1] = __float_as_uint(b); }
#define DEFLDR_F4(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    float a, b, c, d; \
    asm volatile(LDRP S " {%0,%1,%2,%3}, [%4];" : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "l"(p) : "memory"); \
    w[0] = __float_as_uint(a); w[1] = __float_as_uint(b); w[2] = __float_as_uint(c); w[3] = __float_as_uint(d); }
#define DEFLDR_D1(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    double d; asm volatile(LDRP S " %0, [%1];" : "=d"(d) : "l"(p) : "memory"); \
    *(double *)w = d; }
#define DEFLDR_L1(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    unsigned long long l; asm volatile(LDRP S " %0, [%1];" : "=l"(l) : "l"(p) : "memory"); \
    *(unsigned long long *)w = l; }
#define DEFLDR_H2(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    unsigned short h0, h1; \
    asm volatile(LDRP S " {%0,%1}, [%2];" : "=h"(h0), "=h"(h1) : "l"(p) : "memory"); \
    w[0] = h0 | ((uint32_t)h1 << 16); }
#define DEFLDR_H4(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    unsigned short h0, h1, h2, h3; \
    asm volatile(LDRP S " {%0,%1,%2,%3}, [%4];" : "=h"(h0), "=h"(h1), "=h"(h2), "=h"(h3) \
        : "l"(p) : "memory"); \
    w[0] = h0 | ((uint32_t)h1 << 16); w[1] = h2 | ((uint32_t)h3 << 16); }
#define DEFLDR_H8(id, S) static __device__ void ldr_##id(const char *p, uint32_t *w) { \
    unsigned short h0, h1, h2, h3, h4, h5, h6, h7; \
    asm volatile(LDRP S " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" \
        : "=h"(h0), "=h"(h1), "=h"(h2), "=h"(h3), "=h"(h4), "=h"(h5), "=h"(h6), "=h"(h7) \
        : "l"(p) : "memory"); \
    w[0] = h0 | ((uint32_t)h1 << 16); w[1] = h2 | ((uint32_t)h3 << 16); \
    w[2] = h4 | ((uint32_t)h5 << 16); w[3] = h6 | ((uint32_t)h7 << 16); }

#define DEFRED_R1(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    asm volatile(REDP S " [%0], %1;" :: "l"(p), "r"(w[0]) : "memory"); }
#define DEFRED_R2(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    asm volatile(REDP S " [%0], {%1,%2};" :: "l"(p), "r"(w[0]), "r"(w[1]) : "memory"); }
#define DEFRED_R4(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    asm volatile(REDP S " [%0], {%1,%2,%3,%4};" \
        :: "l"(p), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory"); }
#define DEFRED_F1(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    float a = __uint_as_float(w[0]); \
    asm volatile(REDP S " [%0], %1;" :: "l"(p), "f"(a) : "memory"); }
#define DEFRED_F2(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    float a = __uint_as_float(w[0]), b = __uint_as_float(w[1]); \
    asm volatile(REDP S " [%0], {%1,%2};" :: "l"(p), "f"(a), "f"(b) : "memory"); }
#define DEFRED_F4(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    float a = __uint_as_float(w[0]), b = __uint_as_float(w[1]); \
    float c = __uint_as_float(w[2]), d = __uint_as_float(w[3]); \
    asm volatile(REDP S " [%0], {%1,%2,%3,%4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory"); }
#define DEFRED_D1(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    double d = *(const double *)w; \
    asm volatile(REDP S " [%0], %1;" :: "l"(p), "d"(d) : "memory"); }
#define DEFRED_L1(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    unsigned long long l = *(const unsigned long long *)w; \
    asm volatile(REDP S " [%0], %1;" :: "l"(p), "l"(l) : "memory"); }
#define LO(x) ((unsigned short)((x) & 0xffff))
#define HI(x) ((unsigned short)((x) >> 16))
#define DEFRED_H2(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    unsigned short h0 = LO(w[0]), h1 = HI(w[0]); \
    asm volatile(REDP S " [%0], {%1,%2};" :: "l"(p), "h"(h0), "h"(h1) : "memory"); }
#define DEFRED_H4(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    unsigned short h0 = LO(w[0]), h1 = HI(w[0]), h2 = LO(w[1]), h3 = HI(w[1]); \
    asm volatile(REDP S " [%0], {%1,%2,%3,%4};" \
        :: "l"(p), "h"(h0), "h"(h1), "h"(h2), "h"(h3) : "memory"); }
#define DEFRED_H8(id, S) static __device__ void red_##id(char *p, const uint32_t *w) { \
    unsigned short h0 = LO(w[0]), h1 = HI(w[0]), h2 = LO(w[1]), h3 = HI(w[1]); \
    unsigned short h4 = LO(w[2]), h5 = HI(w[2]), h6 = LO(w[3]), h7 = HI(w[3]); \
    asm volatile(REDP S " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" \
        :: "l"(p), "h"(h0), "h"(h1), "h"(h2), "h"(h3), "h"(h4), "h"(h5), "h"(h6), "h"(h7) \
        : "memory"); }

#define X_DEFLDR(id, op, ety, ne, shape, S) DEFLDR_##shape(id, S)
#define X_DEFRED(id, op, ety, ne, shape, S) DEFRED_##shape(id, S)
LDR_TESTS(X_DEFLDR)
RED_TESTS(X_DEFRED)

#define X_ENUM_LDR(id, ...) LDR_##id,
#define X_ENUM_RED(id, ...) RED_##id,
enum { LDR_TESTS(X_ENUM_LDR) NLDR };
enum { RED_TESTS(X_ENUM_RED) NRED };

#define X_CASE_LDR(id, ...) case LDR_##id: ldr_##id(p, w); break;
#define X_CASE_RED(id, ...) case RED_##id: red_##id(p, w); break;

// one slot is SLOT bytes = 8 words
__global__ void ldr_k(const char *mc, uint32_t *out)
{
    for (int t = 0; t < NLDR; t++)
        for (int c = 0; c < NC; c++) {
            const char *p = mc + (t * NC + c) * SLOT;
            uint32_t *w = out + (t * NC + c) * (SLOT / 4);
            switch (t) { LDR_TESTS(X_CASE_LDR) }
        }
}

__global__ void red_k(char *mc, const uint32_t *opnd)
{
    for (int t = 0; t < NRED; t++)
        for (int c = 0; c < NC; c++) {
            char *p = mc + (t * NC + c) * SLOT;
            const uint32_t *w = opnd + (t * NC + c) * (SLOT / 4);
            switch (t) { RED_TESTS(X_CASE_RED) }
        }
}

// ---------------------------------------------------------------------------------------------
// Host code
// ---------------------------------------------------------------------------------------------
struct test_t { const char *s; int op, ety, ne; };
#define X_TAB(id, op, ety, ne, shape, S) { S, op, ety, ne },
static const test_t ldr_t[] = { LDR_TESTS(X_TAB) };
static const test_t red_t[] = { RED_TESTS(X_TAB) };

static int esize(int e)
{
    static const int s[] = { 4, 8, 2, 2, 4, 4, 8, 8, 4, 8 };
    return s[e];
}
static bool is_float(int e) { return e <= E_BF16; }
static bool is_bitwise(int e) { return e >= E_B32; }
static bool is_signed(int e) { return e == E_S32 || e == E_S64; }

// special values, as raw bits
enum { V_QNAN, V_SNAN, V_INF, V_NINF };
static uint64_t special(int e, int v)
{
    static const uint64_t t[4][4] = {            // f32, f64, f16, bf16
        { 0x7fc00000, 0x7ff8000000000000ull, 0x7e00, 0x7fc0 },
        { 0x7fa00000, 0x7ff4000000000000ull, 0x7d00, 0x7fa0 },
        { 0x7f800000, 0x7ff0000000000000ull, 0x7c00, 0x7f80 },
        { 0xff800000, 0xfff0000000000000ull, 0xfc00, 0xff80 } };
    return t[v][e];
}

static bool is_quiet_nan(int e, uint64_t b)
{
    switch (e) {
    case E_F32:  return (b >> 22) & 1;
    case E_F64:  return (b >> 51) & 1;
    case E_F16:  return (b >> 9) & 1;
    default:     return (b >> 6) & 1;
    }
}

static uint64_t enc_num(int e, double v)
{
    switch (e) {
    case E_F32:  { float f = (float)v; uint32_t u; memcpy(&u, &f, 4); return u; }
    case E_F64:  { uint64_t u; memcpy(&u, &v, 8); return u; }
    case E_F16:  return __half_as_ushort(__float2half((float)v));
    case E_BF16: return __bfloat16_as_ushort(__float2bfloat16((float)v));
    case E_U32: case E_B32: return (uint32_t)(uint64_t)v;
    case E_S32:  return (uint32_t)(int32_t)v;
    case E_S64:  return (uint64_t)(int64_t)v;
    default:     return (uint64_t)v;
    }
}

// the operand rank r contributes
static uint64_t operand_bits(int c, int r, int nr, int e)
{
    if (!is_float(e)) {
        if (is_bitwise(e)) return (1ull << r) | 0xF0000;     // distinct bit per rank plus a common one
        if (is_signed(e))  return enc_num(e, r - 3);         // mix of negative and positive
        return enc_num(e, r + 1);
    }
    uint64_t one = enc_num(e, 1), q = special(e, V_QNAN);
    switch (c) {
    case C_FINITE:      return enc_num(e, r + 1);
    case C_NAN_FIRST:   return r == 0 ? q : one;
    case C_NAN_LAST:    return r == nr - 1 ? q : one;
    case C_ALL_NAN:     return q;
    case C_SNAN_FIRST:  return r == 0 ? special(e, V_SNAN) : one;
    default:            return one;
    }
}

// initial value of a red target
static uint64_t red_init(int op, int e, int c)
{
    if (is_float(e)) {
        if (c == C_NAN_INIT) return special(e, V_QNAN);
        return op == OP_ADD ? enc_num(e, 0) : op == OP_MIN ? special(e, V_INF) : special(e, V_NINF);
    }
    bool wide = esize(e) == 8;
    switch (op) {
    case OP_MIN: return e == E_S32 ? 0x7fffffffu : e == E_S64 ? 0x7fffffffffffffffull : wide ? ~0ull : 0xffffffffu;
    case OP_MAX: return e == E_S32 ? 0x80000000u : e == E_S64 ? 0x8000000000000000ull : 0;
    case OP_AND: return wide ? ~0ull : 0xffffffffu;
    default:     return 0;
    }
}

static void put_elem(uint8_t *p, int e, uint64_t bits) { memcpy(p, &bits, esize(e)); }

static uint64_t get_bits(const uint8_t *p, int e)
{
    uint64_t b = 0;
    memcpy(&b, p, esize(e));
    return b;
}

static double to_double(int e, uint64_t b)
{
    switch (e) {
    case E_F32:  { uint32_t u = (uint32_t)b; float f; memcpy(&f, &u, 4); return f; }
    case E_F64:  { double d; memcpy(&d, &b, 8); return d; }
    case E_F16:  { __half_raw r; r.x = (unsigned short)b; return __half2float(__half(r)); }
    case E_BF16: { __nv_bfloat16_raw r; r.x = (unsigned short)b; return __bfloat162float(__nv_bfloat16(r)); }
    case E_S32:  return (int32_t)b;
    case E_S64:  return (int64_t)b;
    default:     return (double)b;
    }
}

// what the finite case must produce
static uint64_t expected_bits(const test_t &t, int nr)
{
    uint64_t acc = 0;
    double dacc = 0;
    for (int r = 0; r < nr; r++) {
        uint64_t b = operand_bits(C_FINITE, r, nr, t.ety);
        double d = to_double(t.ety, b);
        if (r == 0) { acc = b; dacc = d; continue; }
        switch (t.op) {
        case OP_ADD: dacc += d; break;
        case OP_MIN: dacc = d < dacc ? d : dacc; break;
        case OP_MAX: dacc = d > dacc ? d : dacc; break;
        case OP_AND: acc &= b; break;
        case OP_OR:  acc |= b; break;
        case OP_XOR: acc ^= b; break;
        }
    }
    return is_bitwise(t.ety) ? acc : enc_num(t.ety, dacc);
}

static void fill_slot(uint8_t *slot, const test_t &t, int c, int r, int nr)
{
    for (int i = 0; i < t.ne; i++)
        put_elem(slot + i * esize(t.ety), t.ety, operand_bits(c, r, nr, t.ety));
}

struct result_t {
    uint64_t bits;       // element 0
    double v;
    bool nan, snan, lanes_differ, replicas_differ;
};

static result_t decode(const test_t &t, const uint8_t *slot)
{
    result_t r = {};
    r.bits = get_bits(slot, t.ety);
    r.v = to_double(t.ety, r.bits);
    r.nan = is_float(t.ety) && isnan(r.v);
    r.snan = r.nan && !is_quiet_nan(t.ety, r.bits);
    for (int i = 1; i < t.ne; i++)
        if (get_bits(slot + i * esize(t.ety), t.ety) != r.bits) r.lanes_differ = true;
    return r;
}

static const char *fmt(const result_t &r, char *buf, size_t n)
{
    if (r.nan) snprintf(buf, n, "%s", r.snan ? "sNaN" : "NaN");
    else snprintf(buf, n, "%g", r.v);
    return buf;
}

static bool applies(const test_t &t, int c, bool red)
{
    if (!is_float(t.ety)) return c == C_FINITE;
    return red || c != C_NAN_INIT;
}

static void report(const char *kind, bool isred, const test_t &t, const result_t (&res)[NC], int nr, bool verbose)
{
    char name[96], b[NC][16];
    snprintf(name, sizeof name, "%s.%s", kind, t.s);
    uint64_t want = expected_bits(t, nr);
    bool ok = res[C_FINITE].bits == want && !res[C_FINITE].lanes_differ;
    bool any_diff = false, any_snan = false;
    for (int c = 0; c < NC; c++)
        if (applies(t, c, isred)) {
            any_diff |= res[c].replicas_differ || res[c].lanes_differ;
            any_snan |= res[c].snan;
        }

    if (verbose) {
        printf("%s\n", name);
        for (int c = 0; c < NC; c++) {
            if (!applies(t, c, isred)) continue;
            char vb[16];
            printf("    %-12s -> %-8s 0x%0*llx%s%s%s\n", cname[c], fmt(res[c], vb, sizeof vb),
                   2 * esize(t.ety), (unsigned long long)res[c].bits,
                   res[c].lanes_differ ? "  [lanes differ]" : "",
                   res[c].replicas_differ ? "  [replicas differ]" : "",
                   c == C_FINITE ? (ok ? "  ok" : "  MISMATCH") : "");
        }
    }
    if (!is_float(t.ety)) {
        if (!verbose) printf("%-34s finite %s\n", name, ok ? "ok" : "MISMATCH");
        return;
    }
    const char *cls = "mixed";
    if (res[C_NAN_FIRST].nan && res[C_NAN_LAST].nan) cls = "propagates NaN";
    else if (!res[C_NAN_FIRST].nan && !res[C_NAN_LAST].nan &&
             res[C_NAN_FIRST].v == 1 && res[C_NAN_LAST].v == 1) cls = "ignores NaN";
    printf("%s%-34s finite %-8s %-15s first=%s last=%s all=%s snan=%s%s%s%s%s\n",
           verbose ? "  => " : "", verbose ? "" : name, ok ? "ok" : "MISMATCH", cls,
           fmt(res[C_NAN_FIRST], b[0], 16), fmt(res[C_NAN_LAST], b[1], 16),
           fmt(res[C_ALL_NAN], b[2], 16), fmt(res[C_SNAN_FIRST], b[3], 16),
           isred ? " mem=" : "", isred ? fmt(res[C_NAN_INIT], b[4], 16) : "",
           any_snan ? "  [returns sNaN]" : "", any_diff ? "  [lanes/replicas differ]" : "");
}

int main(int argc, char **argv)
{
    bool verbose = argc > 1 && !strcmp(argv[1], "-v");
    int ndev = 0;
    CKR(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { printf("need >= 2 GPUs\n"); return 1; }
    CKD(cuInit(0));
    for (int i = 0; i < ndev; i++) {
        int mcast = 0;
        CKD(cuDeviceGetAttribute(&mcast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, i));
        if (!mcast) { printf("GPU %d: multicast not supported (need NVSwitch, fabric manager, recent driver)\n", i); return 2; }
    }

    CUmulticastObjectProp mp = {};
    mp.numDevices = ndev;
    mp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t mgran = 0, pgran = 0;
    mp.size = 1 << 21;
    CKD(cuMulticastGetGranularity(&mgran, &mp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    CUmemAllocationProp ap = {};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    ap.location.id = 0;
    CKD(cuMemGetAllocationGranularity(&pgran, &ap, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t size = mgran > pgran ? mgran : pgran;
    mp.size = size;
    printf("%d GPUs, multicast granularity %zu, alloc granularity %zu, size %zu\n", ndev, mgran, pgran, size);

    CUmemGenericAllocationHandle mc;
    CKD(cuMulticastCreate(&mc, &mp));
    for (int i = 0; i < ndev; i++) CKD(cuMulticastAddDevice(mc, i));

    CUmemGenericAllocationHandle *mem = (CUmemGenericAllocationHandle *)calloc(ndev, sizeof *mem);
    CUdeviceptr *uc = (CUdeviceptr *)calloc(ndev, sizeof *uc), *mcp = (CUdeviceptr *)calloc(ndev, sizeof *mcp);
    uint32_t **dopnd = (uint32_t **)calloc(ndev, sizeof *dopnd);

    const size_t ldr_bytes = (size_t)NLDR * NC * SLOT, red_bytes = (size_t)NRED * NC * SLOT;
    if (ldr_bytes > RED_BASE || RED_BASE + red_bytes > size) { fprintf(stderr, "slot area too small\n"); return 1; }
    uint8_t *img = (uint8_t *)calloc(1, RED_BASE + red_bytes);
    uint8_t *opnd = (uint8_t *)calloc(1, red_bytes);

    for (int r = 0; r < ndev; r++) {
        CKR(cudaSetDevice(r));
        CKR(cudaFree(0));
        ap.location.id = r;
        CKD(cuMemCreate(&mem[r], size, &ap, 0));
        CKD(cuMulticastBindMem(mc, 0, mem[r], 0, size, 0));
        CUmemAccessDesc acc = {};
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        acc.location.id = r;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CKD(cuMemAddressReserve(&uc[r], size, 0, 0, 0));
        CKD(cuMemMap(uc[r], size, 0, mem[r], 0));
        CKD(cuMemSetAccess(uc[r], size, &acc, 1));
        CKD(cuMemAddressReserve(&mcp[r], size, mgran, 0, 0));
        CKD(cuMemMap(mcp[r], size, 0, mc, 0));
        CKD(cuMemSetAccess(mcp[r], size, &acc, 1));

        memset(img, 0, RED_BASE + red_bytes);
        memset(opnd, 0, red_bytes);
        for (int t = 0; t < NLDR; t++)
            for (int c = 0; c < NC; c++) fill_slot(img + (t * NC + c) * SLOT, ldr_t[t], c, r, ndev);
        for (int t = 0; t < NRED; t++)
            for (int c = 0; c < NC; c++) {
                const test_t &tt = red_t[t];
                fill_slot(opnd + (t * NC + c) * SLOT, tt, c, r, ndev);
                uint8_t *tgt = img + RED_BASE + (t * NC + c) * SLOT;
                for (int i = 0; i < tt.ne; i++)
                    put_elem(tgt + i * esize(tt.ety), tt.ety, red_init(tt.op, tt.ety, c));
            }
        CKR(cudaMemcpy((void *)uc[r], img, RED_BASE + red_bytes, cudaMemcpyHostToDevice));
        CKR(cudaMalloc(&dopnd[r], red_bytes));
        CKR(cudaMemcpy(dopnd[r], opnd, red_bytes, cudaMemcpyHostToDevice));
        CKR(cudaDeviceSynchronize());
    }

    // ---- ld_reduce, issued from GPU 0 ----
    CKR(cudaSetDevice(0));
    uint32_t *dout;
    CKR(cudaMalloc(&dout, ldr_bytes));
    CKR(cudaMemset(dout, 0, ldr_bytes));
    ldr_k<<<1, 1>>>((const char *)mcp[0], dout);
    CKR(cudaGetLastError());
    uint8_t *hout = (uint8_t *)malloc(ldr_bytes);
    CKR(cudaMemcpy(hout, dout, ldr_bytes, cudaMemcpyDeviceToHost));

    printf("\n== multimem.ld_reduce (%d ranks) ==\n", ndev);
    for (int t = 0; t < NLDR; t++) {
        result_t res[NC] = {};
        for (int c = 0; c < NC; c++) res[c] = decode(ldr_t[t], hout + (t * NC + c) * SLOT);
        report("multimem.ld_reduce", false, ldr_t[t], res, ndev, verbose);
    }

    // ---- red: each rank reduces its operand into the shared location, one rank at a time ----
    for (int r = 0; r < ndev; r++) {
        CKR(cudaSetDevice(r));
        red_k<<<1, 1>>>((char *)mcp[r] + RED_BASE, dopnd[r]);
        CKR(cudaGetLastError());
        CKR(cudaDeviceSynchronize());
    }
    uint8_t **back = (uint8_t **)calloc(ndev, sizeof *back);
    for (int r = 0; r < ndev; r++) {
        CKR(cudaSetDevice(r));
        back[r] = (uint8_t *)malloc(red_bytes);
        CKR(cudaMemcpy(back[r], (void *)(uc[r] + RED_BASE), red_bytes, cudaMemcpyDeviceToHost));
    }

    printf("\n== multimem.red (%d ranks; result read back from rank 0's replica) ==\n", ndev);
    for (int t = 0; t < NRED; t++) {
        result_t res[NC] = {};
        for (int c = 0; c < NC; c++) {
            const uint8_t *s0 = back[0] + (t * NC + c) * SLOT;
            res[c] = decode(red_t[t], s0);
            for (int r = 1; r < ndev; r++)
                if (memcmp(s0, back[r] + (t * NC + c) * SLOT, red_t[t].ne * esize(red_t[t].ety)))
                    res[c].replicas_differ = true;
        }
        report("multimem.red", true, red_t[t], res, ndev, verbose);
    }
    return 0;
}
