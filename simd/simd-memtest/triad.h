#ifndef TRIAD_H
#define TRIAD_H

#include "compiler.h"

void triad_ref(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mov(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_rep_movsq(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

/* SSE+ */
void triad_movnti(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_movnti64(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_movntq64(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_movapd128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_movntpd128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_movntdqa128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

/* AVX and AVX-2 */
void triad_vmovapd256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntpd256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vmovntdqa256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vgatherdpd128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vgatherqpd128(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vgatherdpd256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_vgatherqpd256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);
void triad_mvgatherqpd256(size_t n, double s, const double * RESTRICT a, const double * RESTRICT b, double * RESTRICT c);

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

#endif /* TRIAD_H */
