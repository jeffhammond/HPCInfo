// CUDA backend.  Built by nvcc only when NT_CUDA is defined (see Makefile).
#include "nan_common.h"

#ifdef NT_CUDA

#include <cuda_runtime.h>
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

// PTX max/min with and without .NaN (the .NaN forms need sm_80+).
__global__ void ptx_k(const float *a, const float *b, float *r, int n)
{
    int i = threadIdx.x;
    if (i >= n) return;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    float mx, mn, mxn, mnn;
    asm("max.f32 %0, %1, %2;"     : "=f"(mx)  : "f"(a[i]), "f"(b[i]));
    asm("min.f32 %0, %1, %2;"     : "=f"(mn)  : "f"(a[i]), "f"(b[i]));
    asm("max.NaN.f32 %0, %1, %2;" : "=f"(mxn) : "f"(a[i]), "f"(b[i]));
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(mnn) : "f"(a[i]), "f"(b[i]));
    r[0 * n + i] = mx;
    r[1 * n + i] = mn;
    r[2 * n + i] = mxn;
    r[3 * n + i] = mnn;
#endif
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

static void run_ptx(void)
{
    const int n = 2;
    float ha[2], hb[2], hr[4 * 2];
    for (int i = 0; i < n; i++) { ha[i] = (float)nt_a[i]; hb[i] = (float)nt_b[i]; }

    float *da, *db, *dr;
    CK(cudaMalloc(&da, sizeof ha));
    CK(cudaMalloc(&db, sizeof hb));
    CK(cudaMalloc(&dr, sizeof hr));
    CK(cudaMemcpy(da, ha, sizeof ha, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb, sizeof hb, cudaMemcpyHostToDevice));
    ptx_k<<<1, n>>>(da, db, dr, n);
    CK(cudaGetLastError());
    CK(cudaMemcpy(hr, dr, sizeof hr, cudaMemcpyDeviceToHost));
    cudaFree(da); cudaFree(db); cudaFree(dr);

    static const char *label[4] = {"PTX max.f32", "PTX min.f32", "PTX max.NaN.f32", "PTX min.NaN.f32"};
    static const char *op[4]    = {"max", "min", "max", "min"};
    for (int k = 0; k < 4; k++)
        nt_record(label[k], op[k], (double)hr[k * n + 0], (double)hr[k * n + 1]);
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
        char note[256];
        snprintf(note, sizeof note, "CUDA device %d: %s (sm_%d%d)", d, prop.name, prop.major, prop.minor);
        nt_note(note);

        run_type<double>("double");
        run_type<float>("float");
        if (prop.major >= 8) run_ptx();
        else nt_skip("PTX max/min.NaN.f32", "needs sm_80 or newer");
    }
}

#endif /* NT_CUDA */
