#ifndef COPY_H
#define COPY_H

#include "compiler.h"

void copy_ref(size_t n, const double * RESTRICT a, double * RESTRICT b);

/* aarch64 */
void copy_vld1q(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vld1q_x2(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vld1q_x3(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vld1q_x4(size_t n, const double * RESTRICT a, double * RESTRICT b);

/* x86_64 */
void copy_mov(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_rep_movsq(size_t n, const double * RESTRICT a, double * RESTRICT b);

/* SSE+ */
void copy_movnti(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_movnti64(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_movntq64(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_movapd128(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_movntpd128(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_movntdqa128(size_t n, const double * RESTRICT a, double * RESTRICT b);

/* AVX and AVX-2 */
void copy_vmovapd256(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vmovntpd256(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vmovntdqa256(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vgatherdpd128(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vgatherqpd128(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vgatherdpd256(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vgatherqpd256(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_mvgatherqpd256(size_t n, const double * RESTRICT a, double * RESTRICT b);

/* AVX-512F */
void copy_vmovapd512(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vmovupd512(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_mvmovapd512(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_mvmovupd512(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vmovntdqa512(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vmovntpd512(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vGSdpd512(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_mvGSdpd512(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_vGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b);
void copy_mvGSqpd512(size_t n, const double * RESTRICT a, double * RESTRICT b);

#endif /* COPY_H */
