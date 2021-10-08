#ifndef COMPILER_H
#define COMPILER_H

#ifdef __cplusplus
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define RESTRICT __restrict__
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define RESTRICT restrict
#endif

#if (__STDC_VERSION__ >= 199901L) || (__cplusplus >= 201103L)
#ifdef _OPENMP
#define PRAGMA(x) _Pragma(#x)
#endif
#else
#warning Your compiler does not support C99/C++11 _Pragma.
#define PRAGMA(x)
#endif

#if defined(__INTEL_COMPILER)
#define ASSUME(a) __assume(a)
#define UNROLL_AND_JAM(n) PRAGMA(unroll_and_jam(n))
#else
#define ASSUME(a)
#define UNROLL_AND_JAM(n)
#endif

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__INTEL_COMPILER)
typedef int64_t __int64;
#endif

#if defined(__GNUC__) || defined(__INTEL_COMPILER)
#define HAS_GNU_EXTENDED_ASM 1
#else
#define HAS_GNU_EXTENDED_ASM 0
#endif

#ifdef __x86_64__
#include "immintrin.h"
#if !(defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1700))
#ifndef _MM_UPCONV_PD_NONE
#define _MM_UPCONV_PD_NONE 0
#endif
#ifndef _MM_DOWNCONV_PD_NONE
#define _MM_DOWNCONV_PD_NONE 0
#endif
#ifndef _MM_HINT_NONE
#define _MM_HINT_NONE 0
#endif
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#define OMP_PARALLEL_FOR PRAGMA(omp parallel for)
#else
#define OMP_PARALLEL_FOR
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_max_threads() { return 1; }
static inline double omp_get_wtime() { return 0.0; }
#endif

#endif /* COMPILER_H */
