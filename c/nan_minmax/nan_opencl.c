#include "nan_common.h"

#ifdef NT_OPENCL

#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* T is float or double, chosen with -D at program build time. */
static const char *kernel_src =
    "#ifdef USE_FP64\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "#endif\n"
    "__kernel void ops(__global const T *a, __global const T *b, __global T *r, const int n)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    T x = a[i], y = b[i];\n"
    "    r[0 * n + i] = x + y;\n"
    "    r[1 * n + i] = x * y;\n"
    "    r[2 * n + i] = fmax(x, y);\n"
    "    r[3 * n + i] = fmin(x, y);\n"
    "    r[4 * n + i] = max(x, y);\n"
    "    r[5 * n + i] = min(x, y);\n"
    "    r[6 * n + i] = (x > y) ? x : y;\n"
    "    r[7 * n + i] = (x < y) ? x : y;\n"
    "}\n";

#define NOPS 8
static const char *label[NOPS] = {"x + y", "x * y", "fmax()", "fmin()",
                                  "max()", "min()", "x>y?x:y", "x<y?x:y"};
static const char *opname[NOPS] = {"add", "mul", "max", "min", "max", "min", "max", "min"};

/* Run the kernel on 2 work-items; abytes is the size of each input array. */
static int exec(cl_context ctx, cl_command_queue q, cl_kernel k,
                const void *a, const void *b, void *r, size_t abytes, size_t rbytes)
{
    cl_int err, n = 2;
    cl_mem ma = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, abytes, (void *)a, &err);
    cl_mem mb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, abytes, (void *)b, &err);
    cl_mem mr = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, rbytes, NULL, &err);
    size_t gsz = 2;

    err  = clSetKernelArg(k, 0, sizeof ma, &ma);
    err |= clSetKernelArg(k, 1, sizeof mb, &mb);
    err |= clSetKernelArg(k, 2, sizeof mr, &mr);
    err |= clSetKernelArg(k, 3, sizeof n, &n);
    err |= clEnqueueNDRangeKernel(q, k, 1, NULL, &gsz, NULL, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(q, mr, CL_TRUE, 0, rbytes, r, 0, NULL, NULL);

    clReleaseMemObject(ma);
    clReleaseMemObject(mb);
    clReleaseMemObject(mr);
    return err == CL_SUCCESS;
}

static void run_type(cl_context ctx, cl_command_queue q, cl_device_id dev, int is_double)
{
    const char *tname = is_double ? "double" : "float";
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    err = clBuildProgram(prog, 1, &dev, is_double ? "-DT=double -DUSE_FP64" : "-DT=float", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[2048] = "";
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, NULL);
        printf("  OpenCL %s build failed (%d): %s\n", tname, (int)err, log);
        clReleaseProgram(prog);
        return;
    }
    cl_kernel k = clCreateKernel(prog, "ops", &err);

    double res[NOPS][2];
    int ok;
    if (is_double) {
        double a[2], b[2], r[NOPS * 2];
        for (int i = 0; i < 2; i++) { a[i] = nt_a[i]; b[i] = nt_b[i]; }
        ok = exec(ctx, q, k, a, b, r, sizeof a, sizeof r);
        for (int j = 0; j < NOPS; j++) { res[j][0] = r[j * 2]; res[j][1] = r[j * 2 + 1]; }
    } else {
        float a[2], b[2], r[NOPS * 2];
        for (int i = 0; i < 2; i++) { a[i] = (float)nt_a[i]; b[i] = (float)nt_b[i]; }
        ok = exec(ctx, q, k, a, b, r, sizeof a, sizeof r);
        for (int j = 0; j < NOPS; j++) { res[j][0] = r[j * 2]; res[j][1] = r[j * 2 + 1]; }
    }

    if (!ok) {
        printf("  OpenCL %s kernel execution failed\n", tname);
    } else {
        for (int j = 0; j < NOPS; j++) {
            char impl[64];
            snprintf(impl, sizeof impl, "OpenCL %s %s", tname, label[j]);
            nt_record(impl, opname[j], res[j][0], res[j][1]);
        }
    }
    clReleaseKernel(k);
    clReleaseProgram(prog);
}

void nt_run_opencl(void)
{
    cl_uint np = 0;
    if (clGetPlatformIDs(0, NULL, &np) != CL_SUCCESS || np == 0) {
        nt_skip("OpenCL", "no platforms");
        return;
    }
    cl_platform_id *plat = malloc(np * sizeof *plat);
    clGetPlatformIDs(np, plat, NULL);

    int ndev_total = 0;
    for (cl_uint p = 0; p < np; p++) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(plat[p], CL_DEVICE_TYPE_ALL, 0, NULL, &nd) != CL_SUCCESS || nd == 0)
            continue;
        ndev_total += (int)nd;
        cl_device_id *dev = malloc(nd * sizeof *dev);
        clGetDeviceIDs(plat[p], CL_DEVICE_TYPE_ALL, nd, dev, NULL);

        for (cl_uint d = 0; d < nd; d++) {
            char name[256] = "", ext[8192] = "";
            clGetDeviceInfo(dev[d], CL_DEVICE_NAME, sizeof name - 1, name, NULL);
            clGetDeviceInfo(dev[d], CL_DEVICE_EXTENSIONS, sizeof ext - 1, ext, NULL);

            char note[320];
            snprintf(note, sizeof note, "OpenCL device: %s", name);
            nt_note(note);

            cl_int err;
            cl_context ctx = clCreateContext(NULL, 1, &dev[d], NULL, NULL, &err);
            if (err != CL_SUCCESS) { nt_skip("OpenCL", "clCreateContext failed"); continue; }
            cl_command_queue q = clCreateCommandQueue(ctx, dev[d], 0, &err);
            if (err != CL_SUCCESS) {
                nt_skip("OpenCL", "clCreateCommandQueue failed");
                clReleaseContext(ctx);
                continue;
            }

            run_type(ctx, q, dev[d], 0);
            if (strstr(ext, "cl_khr_fp64")) run_type(ctx, q, dev[d], 1);
            else nt_skip("OpenCL double", "device lacks cl_khr_fp64");

            clReleaseCommandQueue(q);
            clReleaseContext(ctx);
        }
        free(dev);
    }
    free(plat);
    if (ndev_total == 0)
        nt_skip("OpenCL", "platform(s) found but no devices");
}

#endif /* NT_OPENCL */
