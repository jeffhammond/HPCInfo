#ifndef UTIL_H
#define UTIL_H

#include "compiler.h"

#ifdef __AVX512F__
static const size_t alignment = 2097152; // 2M
#else
static const size_t alignment = 4096;    // 4K
#endif

void * mymalloc(size_t bytes);
size_t compare_doubles(size_t n, const double * RESTRICT x, const double * RESTRICT y);
size_t compare_doubles_stride(size_t n, const double * RESTRICT x, const double * RESTRICT y, int stride);
size_t compare_doubles_stride_holes(size_t n, const double * RESTRICT x, const double * RESTRICT y, int stride, double val);
void init_doubles(size_t n, double * RESTRICT x);
void set_doubles(size_t n, double value, double * RESTRICT x);
void print_doubles_1(size_t n, const double * RESTRICT x);
void print_doubles_2(size_t n, const double * RESTRICT x, const double * RESTRICT y);
void print_compare_doubles_stride_holes(size_t n, const double * RESTRICT x, const double * RESTRICT y, int stride, double val);

#endif /* UTIL_H */
