#ifndef SETUP_H
#define SETUP_H

#include "copy.h"
#include "stride.h"

#define MAXTEST 64

/* copy */
double testtime[MAXTEST] = {0};
typedef void (*copyfn)(size_t, const double * RESTRICT, double * RESTRICT);
copyfn testfns[MAXTEST] = {0};
const char* testname[MAXTEST] = {0};

/* strided */
double testtime2[MAXTEST] = {0};
typedef void (*stridefn)(size_t, const double * RESTRICT, double * RESTRICT, unsigned);
stridefn testfns2[MAXTEST] = {0};
const char* testname2[MAXTEST] = {0};

int setup(void)
{
    int i = 0;
    {
        /* Reference */
        testfns[i]  = copy_ref;
        testname[i] = "Reference";
        i++;
        /* MOV */
        testfns[i]  = copy_mov;
        testname[i] = "mov";
        i++;
        /* REP_MOVSQ */
        testfns[i]  = copy_rep_movsq;
        testname[i] = "rep_movsq";
        i++;
    }
#ifdef __SSE2__
    {
        /* MOVNTI */
        testfns[i]  = copy_movnti;
        testname[i] = "movnti";
        i++;
#if 0 /* BROKEN */
        /* MOVNTQ */
        testfns[i]  = copy_movntq;
        testname[i] = "movntq";
        i++;
#endif
#ifdef __INTEL_COMPILER
        /* MOVNTI64 */
        testfns[i]  = copy_movnti64;
        testname[i] = "movnti64";
        i++;
        /* MOVNTQ64 */
        testfns[i]  = copy_movntq64;
        testname[i] = "movntq64";
        i++;
#endif
        /* MOVAPD128 */
        testfns[i]  = copy_movapd128;
        testname[i] = "movapd128";
        i++;
        /* MOVNTPD128 */
        testfns[i]  = copy_movntpd128;
        testname[i] = "movntpd128";
        i++;
    }
#endif
#ifdef __SSE4_1__
    {
        /* MOVNTDQA128 */
        testfns[i]  = copy_movntdqa128;
        testname[i] = "movntdqa128";
        i++;
    }
#endif
#ifdef __AVX__
    {
        /* VMOVAPD256 */
        testfns[i]  = copy_vmovapd256;
        testname[i] = "vmovapd256";
        i++;
        /* VMOVNTPD256 */
        testfns[i]  = copy_vmovntpd256;
        testname[i] = "vmovntpd256";
        i++;
    }
#endif
#ifdef __AVX2__
    {
        /* VMOVNTDQA256 */
        testfns[i]  = copy_vmovntdqa256;
        testname[i] = "vmovntdqa256";
        i++;
        /* VGATHERDPD128 */
        testfns[i]  = copy_vgatherdpd128;
        testname[i] = "vgatherdpd128";
        i++;
        /* VGATHERQPD128 */
        testfns[i]  = copy_vgatherqpd128;
        testname[i] = "vgatherqpd128";
        i++;
        /* VGATHERDPD256 */
        testfns[i]  = copy_vgatherdpd256;
        testname[i] = "vgatherdpd256";
        i++;
        /* VGATHERQPD256 */
        testfns[i]  = copy_vgatherqpd256;
        testname[i] = "vgatherqpd256";
        i++;
        /* MVGATHERQPD256 */
        testfns[i]  = copy_mvgatherqpd256;
        testname[i] = "mvgatherqpd256";
        i++;
    }
#endif
#ifdef __AVX512F__
    {
        /* VMOVAPD512 */
        testfns[i]  = copy_vmovapd512;
        testname[i] = "vmovapd512";
        i++;
        /* VMOVUPD512 */
        testfns[i]  = copy_vmovupd512;
        testname[i] = "vmovupd512";
        i++;
        /* MVMOVAPD512 */
        testfns[i]  = copy_mvmovapd512;
        testname[i] = "mvmovapd512";
        i++;
        /* MVMOVUPD512 */
        testfns[i]  = copy_mvmovupd512;
        testname[i] = "mvmovupd512";
        i++;
        /* VMOVNTPD512 */
        testfns[i]  = copy_vmovntpd512;
        testname[i] = "vmovntpd512";
        i++;
        /* VMOVNTDQA512 */
        testfns[i]  = copy_vmovntdqa512;
        testname[i] = "vmovntdqa512";
        i++;
        /* VGSDPD512 */
        testfns[i]  = copy_vGSdpd512;
        testname[i] = "vGSdpd512";
        i++;
        /* MVGSDPD512 */
        testfns[i]  = copy_mvGSdpd512;
        testname[i] = "mvGSdpd512";
        i++;
        /* VGSQPD512 */
        testfns[i]  = copy_vGSqpd512;
        testname[i] = "vGSqpd512";
        i++;
        /* MVGSQPD512 */
        testfns[i]  = copy_mvGSqpd512;
        testname[i] = "mvGSqpd512";
        i++;
    }
#endif
    return i;
}

int setup_stride(void)
{
    int i = 0;
    {
        /* Reference */
        testfns2[i]  = stride_ref;
        testname2[i] = "Reference";
        i++;
        /* MOV */
        testfns2[i]  = stride_mov;
        testname2[i] = "mov";
        i++;
    }
#ifdef __SSE2__
    {
        /* MOVNTI */
        testfns2[i]  = stride_movnti;
        testname2[i] = "movnti";
        i++;
#ifdef __INTEL_COMPILER
        /* MOVNTI64 */
        testfns2[i]  = stride_movnti64;
        testname2[i] = "movnti64";
        i++;
        /* MOVNTQ64 */
        testfns2[i]  = stride_movntq64;
        testname2[i] = "movntq64";
        i++;
#endif
    }
#endif
#ifdef __SSE4_1__
    {
    }
#endif
#ifdef __AVX__
    {
    }
#endif
#ifdef __AVX2__
    {
        /* VGATHERDPD128 */
        testfns2[i]  = stride_vgatherdpd128;
        testname2[i] = "vgatherdpd128";
        i++;
        /* VGATHERQPD128 */
        testfns2[i]  = stride_vgatherqpd128;
        testname2[i] = "vgatherqpd128";
        i++;
        /* VGATHERDPD256 */
        testfns2[i]  = stride_vgatherdpd256;
        testname2[i] = "vgatherdpd256";
        i++;
        /* VGATHERQPD256 */
        testfns2[i]  = stride_vgatherqpd256;
        testname2[i] = "vgatherqpd256";
        i++;
#if 0
        /* MVGATHERQPD256 */
        testfns2[i]  = stride_mvgatherqpd256;
        testname2[i] = "mvgatherqpd256";
        i++;
#endif
    }
#endif
#ifdef __AVX512F__
    {
        /* MVMOVAPD512 */
        testfns2[i]  = stride_mvmovapd512;
        testname2[i] = "mvmovapd512";
        i++;
        /* MVMOVUPD512 */
        testfns2[i]  = stride_mvmovupd512;
        testname2[i] = "mvmovupd512";
        i++;
        /* VGSDPD512 */
        testfns2[i]  = stride_vGSdpd512;
        testname2[i] = "vGSdpd512";
        i++;
        /* VGSQPD512 */
        testfns2[i]  = stride_vGSqpd512;
        testname2[i] = "vGSqpd512";
        i++;
        /* MVGSDPD512 */
        testfns2[i]  = stride_mvGSdpd512;
        testname2[i] = "mvGSdpd512";
        i++;
        /* MVGSQPD512 */
        testfns2[i]  = stride_mvGSqpd512;
        testname2[i] = "mvGSqpd512";
        i++;
    }
#endif
    return i;
}

#endif /* SETUP_H */
