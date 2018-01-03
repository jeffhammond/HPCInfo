#ifndef STRIDE_H
#define STRIDE_H

#include "compiler.h"

void stride_ref(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_mov(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);

/* SSE+ */
void stride_movnti(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_movnti64(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_movntq64(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);

/* AVX and AVX-2 */
void stride_vgatherdpd128(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_vgatherqpd128(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_vgatherdpd256(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_vgatherqpd256(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);

/* AVX-512F */
void stride_mvmovapd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_mvmovupd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_vGSdpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_vGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_mvGSdpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);
void stride_mvGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);

/* AVX-512PF */
void stride_vPFGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b, unsigned s);

#endif /* STRIDE_H */
