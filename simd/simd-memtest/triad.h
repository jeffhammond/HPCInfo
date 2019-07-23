#ifndef TRIAD_H
#define TRIAD_H

#include "compiler.h"

void triad_ref(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_official(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

/* SSE+ */
void triad_movapd128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_movntpd128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_movntdq128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

/* 128-bit w/ FMA */
void triad_movapd128fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_movntpd128fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_movntdq128fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

/* AVX and AVX-2 */
void triad_vmovapd256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntpd256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntdqa256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

/* 256-bit w/ FMA */
void triad_vmovapd256fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntpd256fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntdqa256fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

/* AVX-512F */
void triad_vmovapd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovupd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mvmovapd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mvmovupd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntdqa512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vGSdpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mvGSdpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vGSqpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mvGSqpd512(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

/* 512-bit w/ FMA */
void triad_vmovapd512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovupd512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mvmovapd512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mvmovupd512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntdqa512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntpd512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vGSdpd512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mvGSdpd512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vGSqpd512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mvGSqpd512fma(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

#endif /* TRIAD_H */
