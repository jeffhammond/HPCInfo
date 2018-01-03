#include "compiler.h"
#include "util.h"

void * mymalloc(size_t bytes)
{
    void * ptr = NULL;
    int rc = posix_memalign(&ptr, alignment, bytes);
    if (rc != 0 || ptr == NULL) abort();
    return ptr;
}

size_t compare_doubles(size_t n, const double * RESTRICT x, const double * RESTRICT y)
{
    size_t errors = 0;
#pragma omp parallel for reduction(+:errors)
    for (size_t i=0; i<n; i++) {
        if (x[i] != y[i]) errors++;
    }
    return errors;
}

size_t compare_doubles_stride(size_t n, const double * RESTRICT x, const double * RESTRICT y, int stride)
{
    size_t errors = 0;
#pragma omp parallel for reduction(+:errors)
    for (size_t i=0; i<n; i+=stride) {
        if (x[i] != y[i]) errors++;
    }
    return errors;
}

size_t compare_doubles_stride_holes(size_t n, const double * RESTRICT x, const double * RESTRICT y, int stride, double val)
{
    size_t errors = 0;
#pragma omp parallel for reduction(+:errors)
    for (size_t i=0; i<n; i+=stride) {
        /* check the part that is copied */
        if (y[i] != x[i]) errors++;
        /* between the strides, elements should not change */
        for (int s=1; s<stride && i+s<n; s++) {
            if (y[i+s] != val) errors++;
        }
    }
    return errors;
}

void init_doubles(size_t n, double * RESTRICT x)
{
#pragma omp parallel for
    for (size_t i=0; i<n; i++) {
        x[i] = (double)i;
    }
}

void set_doubles(size_t n, double value, double * RESTRICT x)
{
#pragma omp parallel for
    for (size_t i=0; i<n; i++) {
        x[i] = value;
    }
}

void print_doubles_1(size_t n, const double * RESTRICT x)
{
    for (size_t i=0; i<n; i++) {
        printf("%zu %lf\n", i, x[i]);
    }
    fflush(stdout);
}

void print_doubles_2(size_t n, const double * RESTRICT x, const double * RESTRICT y)
{
    for (size_t i=0; i<n; i++) {
        printf("%zu %lf %lf\n", i, x[i], y[i]);
    }
    fflush(stdout);
}

void print_compare_doubles_stride_holes(size_t n, const double * RESTRICT x, const double * RESTRICT y, int stride, double val)
{
    for (size_t i=0; i<n; i+=stride) {
        printf("%zu %lf %lf %s\n", i, y[i], x[i], (y[i]==x[i]) ? "" : "ERROR");
        for (int s=1; s<stride && i+s<n; s++) {
            printf("%zu %lf %lf %s\n", i+s, y[i+s], val, (y[i+s]==val) ? "" : "ERROR");
        }
    }
    fflush(stdout);
}
