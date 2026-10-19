// CUDA backend.  Built by nvcc only when NT_CUDA is defined (see Makefile).
#include "nan_common.h"

#ifdef NT_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>

#define CK(call) do {                                                       \
        cudaError_t e_ = (call);                                            \
        if (e_ != cudaSuccess) {                                            \
            printf("  CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_),    \
                   __FILE__, __LINE__);                                     \
            return;                                                         \
        }                                                                   \
    } while (0)

// One thread per operand order: thread 0 sees (1,NaN), thread 1 sees (NaN,1).
template <typename T>
__global__ void ops_k(const T *a, const T *b, T *r, int n)
{
    int i = threadIdx.x;
    if (i >= n) return;
    T x = a[i], y = b[i];
    r[0 * n + i] = x + y;
    r[1 * n + i] = x * y;
    r[2 * n + i] = fmax(x, y);
    r[3 * n + i] = fmin(x, y);
    r[4 * n + i] = (x > y) ? x : y;
    r[5 * n + i] = (x < y) ? x : y;
}

// Direct PTX.  add/mul/fma have no .NaN modifier (ptxas rejects it), so the
// NaN-selectable forms exist only for min/max: .NaN on f32/f16/bf16 (sm_80+).
// There is no min/max.NaN.f64.  f16/bf16 values travel as .b16 bit patterns.
#define PTX_NRES 20
__global__ void ptx_k(const float *a, const float *b, double *r, int n)
{
    int i = threadIdx.x;
    if (i >= n) return;
    double xd = a[i], yd = b[i], d;
    float  xf = a[i], yf = b[i], f;
    int k = 0;
#define R(v) r[(k++) * n + i] = (v)
    asm("add.rn.f64 %0, %1, %2;"       : "=d"(d) : "d"(xd), "d"(yd));      R(d);
    asm("mul.rn.f64 %0, %1, %2;"       : "=d"(d) : "d"(xd), "d"(yd));      R(d);
    asm("fma.rn.f64 %0, %1, %2, %3;"   : "=d"(d) : "d"(xd), "d"(yd), "d"(0.0)); R(d);
    asm("max.f64 %0, %1, %2;"          : "=d"(d) : "d"(xd), "d"(yd));      R(d);
    asm("min.f64 %0, %1, %2;"          : "=d"(d) : "d"(xd), "d"(yd));      R(d);
    asm("add.rn.f32 %0, %1, %2;"       : "=f"(f) : "f"(xf), "f"(yf));      R(f);
    asm("mul.rn.f32 %0, %1, %2;"       : "=f"(f) : "f"(xf), "f"(yf));      R(f);
    asm("fma.rn.f32 %0, %1, %2, %3;"   : "=f"(f) : "f"(xf), "f"(yf), "f"(0.0f)); R(f);
    asm("max.f32 %0, %1, %2;"          : "=f"(f) : "f"(xf), "f"(yf));      R(f);
    asm("min.f32 %0, %1, %2;"          : "=f"(f) : "f"(xf), "f"(yf));      R(f);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("max.NaN.f32 %0, %1, %2;"      : "=f"(f) : "f"(xf), "f"(yf));      R(f);
    asm("min.NaN.f32 %0, %1, %2;"      : "=f"(f) : "f"(xf), "f"(yf));      R(f);
    unsigned short hx = __half_as_ushort(__float2half_rn(xf)), hy = __half_as_ushort(__float2half_rn(yf)), h;
    asm("max.f16 %0, %1, %2;"          : "=h"(h) : "h"(hx), "h"(hy));      R(__half2float(__ushort_as_half(h)));
    asm("min.f16 %0, %1, %2;"          : "=h"(h) : "h"(hx), "h"(hy));      R(__half2float(__ushort_as_half(h)));
    asm("max.NaN.f16 %0, %1, %2;"      : "=h"(h) : "h"(hx), "h"(hy));      R(__half2float(__ushort_as_half(h)));
    asm("min.NaN.f16 %0, %1, %2;"      : "=h"(h) : "h"(hx), "h"(hy));      R(__half2float(__ushort_as_half(h)));
    unsigned short bx = __bfloat16_as_ushort(__float2bfloat16_rn(xf)), by = __bfloat16_as_ushort(__float2bfloat16_rn(yf));
    asm("max.bf16 %0, %1, %2;"         : "=h"(h) : "h"(bx), "h"(by));      R(__bfloat162float(__ushort_as_bfloat16(h)));
    asm("min.bf16 %0, %1, %2;"         : "=h"(h) : "h"(bx), "h"(by));      R(__bfloat162float(__ushort_as_bfloat16(h)));
    asm("max.NaN.bf16 %0, %1, %2;"     : "=h"(h) : "h"(bx), "h"(by));      R(__bfloat162float(__ushort_as_bfloat16(h)));
    asm("min.NaN.bf16 %0, %1, %2;"     : "=h"(h) : "h"(bx), "h"(by));      R(__bfloat162float(__ushort_as_bfloat16(h)));
#endif
#undef R
}

template <typename T>
static void run_type(const char *tname)
{
    const int n = 2;
    T ha[2], hb[2], hr[6 * 2];
    for (int i = 0; i < n; i++) { ha[i] = (T)nt_a[i]; hb[i] = (T)nt_b[i]; }

    T *da, *db, *dr;
    CK(cudaMalloc(&da, sizeof ha));
    CK(cudaMalloc(&db, sizeof hb));
    CK(cudaMalloc(&dr, sizeof hr));
    CK(cudaMemcpy(da, ha, sizeof ha, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb, sizeof hb, cudaMemcpyHostToDevice));
    ops_k<T><<<1, n>>>(da, db, dr, n);
    CK(cudaGetLastError());
    CK(cudaMemcpy(hr, dr, sizeof hr, cudaMemcpyDeviceToHost));
    cudaFree(da); cudaFree(db); cudaFree(dr);

    static const char *label[6] = {"x + y", "x * y", "fmax()", "fmin()", "x>y?x:y", "x<y?x:y"};
    static const char *op[6]    = {"add", "mul", "max", "min", "max", "min"};
    for (int k = 0; k < 6; k++) {
        char impl[64];
        snprintf(impl, sizeof impl, "CUDA %s %s", tname, label[k]);
        nt_record(impl, op[k], (double)hr[k * n + 0], (double)hr[k * n + 1]);
    }
}

static void run_ptx(bool sm80)
{
    const int n = 2;
    float ha[2], hb[2];
    double hr[PTX_NRES * 2] = {0};
    for (int i = 0; i < n; i++) { ha[i] = (float)nt_a[i]; hb[i] = (float)nt_b[i]; }

    float *da, *db; double *dr;
    CK(cudaMalloc(&da, sizeof ha));
    CK(cudaMalloc(&db, sizeof hb));
    CK(cudaMalloc(&dr, sizeof hr));
    CK(cudaMemcpy(da, ha, sizeof ha, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb, sizeof hb, cudaMemcpyHostToDevice));
    ptx_k<<<1, n>>>(da, db, dr, n);
    CK(cudaGetLastError());
    CK(cudaMemcpy(hr, dr, sizeof hr, cudaMemcpyDeviceToHost));
    cudaFree(da); cudaFree(db); cudaFree(dr);

    // fma(x,y,0) is recorded as "mul": a NaN in either factor must give NaN
    static const struct { const char *impl, *op; } t[PTX_NRES] = {
        {"PTX add.rn.f64", "add"},      {"PTX mul.rn.f64", "mul"},
        {"PTX fma.rn.f64 (x*y+0)", "mul"},
        {"PTX max.f64", "max"},         {"PTX min.f64", "min"},
        {"PTX add.rn.f32", "add"},      {"PTX mul.rn.f32", "mul"},
        {"PTX fma.rn.f32 (x*y+0)", "mul"},
        {"PTX max.f32", "max"},         {"PTX min.f32", "min"},
        {"PTX max.NaN.f32", "max"},     {"PTX min.NaN.f32", "min"},
        {"PTX max.f16", "max"},         {"PTX min.f16", "min"},
        {"PTX max.NaN.f16", "max"},     {"PTX min.NaN.f16", "min"},
        {"PTX max.bf16", "max"},        {"PTX min.bf16", "min"},
        {"PTX max.NaN.bf16", "max"},    {"PTX min.NaN.bf16", "min"},
    };
    // the first 10 need no particular arch; the rest need sm_80
    for (int k = 0; k < (sm80 ? PTX_NRES : 10); k++)
        nt_record(t[k].impl, t[k].op, hr[k * n + 0], hr[k * n + 1]);
}

extern "C" void nt_run_cuda(void)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        nt_skip("CUDA", "no CUDA device");
        return;
    }
    for (int d = 0; d < count; d++) {
        cudaDeviceProp prop;
        CK(cudaGetDeviceProperties(&prop, d));
        CK(cudaSetDevice(d));
        char note[512];
        snprintf(note, sizeof note, "CUDA device %d: %s (sm_%d%d)", d, prop.name, prop.major, prop.minor);
        nt_note(note);

        run_type<double>("double");
        run_type<float>("float");
        run_ptx(prop.major >= 8);
        if (prop.major < 8) nt_skip("PTX min/max .NaN, f16, bf16", "needs sm_80 or newer");
    }
}

#endif /* NT_CUDA */
