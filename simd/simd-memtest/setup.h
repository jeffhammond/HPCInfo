#ifndef SETUP_H
#define SETUP_H

#include "copy.h"
#include "triad.h"
#include "stride.h"

#define MAXTEST 64

/* copy */
double testtime0[MAXTEST] = {0};
typedef void (*copyfn)(size_t, const double * RESTRICT, double * RESTRICT);
copyfn testfns0[MAXTEST] = {0};
const char* testname0[MAXTEST] = {0};

/* triad */
double testtime1[MAXTEST] = {0};
typedef void (*triadfn)(size_t, double, const double * RESTRICT, const double * RESTRICT, double * RESTRICT);
triadfn testfns1[MAXTEST] = {0};
const char* testname1[MAXTEST] = {0};

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
        testfns0[i]  = copy_ref;
        testname0[i] = "Reference";
        i++;
        /* MOV */
        testfns0[i]  = copy_mov;
        testname0[i] = "mov";
        i++;
        /* REP_MOVSQ */
        testfns0[i]  = copy_rep_movsq;
        testname0[i] = "rep_movsq";
        i++;
    }
#ifdef __SSE2__
    {
        /* MOVNTI */
        testfns0[i]  = copy_movnti;
        testname0[i] = "movnti";
        i++;
#if 0 /* BROKEN */
        /* MOVNTQ */
        testfns0[i]  = copy_movntq;
        testname0[i] = "movntq";
        i++;
#endif
#ifdef __INTEL_COMPILER
        /* MOVNTI64 */
        testfns0[i]  = copy_movnti64;
        testname0[i] = "movnti64";
        i++;
        /* MOVNTQ64 */
        testfns0[i]  = copy_movntq64;
        testname0[i] = "movntq64";
        i++;
#endif
        /* MOVAPD128 */
        testfns0[i]  = copy_movapd128;
        testname0[i] = "movapd128";
        i++;
        /* MOVNTPD128 */
        testfns0[i]  = copy_movntpd128;
        testname0[i] = "movntpd128";
        i++;
    }
#endif
#ifdef __SSE4_1__
    {
        /* MOVNTDQA128 */
        testfns0[i]  = copy_movntdqa128;
        testname0[i] = "movntdqa128";
        i++;
    }
#endif
#ifdef __AVX__
    {
        /* VMOVAPD256 */
        testfns0[i]  = copy_vmovapd256;
        testname0[i] = "vmovapd256";
        i++;
        /* VMOVNTPD256 */
        testfns0[i]  = copy_vmovntpd256;
        testname0[i] = "vmovntpd256";
        i++;
    }
#endif
#ifdef __AVX2__
    {
        /* VMOVNTDQA256 */
        testfns0[i]  = copy_vmovntdqa256;
        testname0[i] = "vmovntdqa256";
        i++;
        /* VGATHERDPD128 */
        testfns0[i]  = copy_vgatherdpd128;
        testname0[i] = "vgatherdpd128";
        i++;
        /* VGATHERQPD128 */
        testfns0[i]  = copy_vgatherqpd128;
        testname0[i] = "vgatherqpd128";
        i++;
        /* VGATHERDPD256 */
        testfns0[i]  = copy_vgatherdpd256;
        testname0[i] = "vgatherdpd256";
        i++;
        /* VGATHERQPD256 */
        testfns0[i]  = copy_vgatherqpd256;
        testname0[i] = "vgatherqpd256";
        i++;
        /* MVGATHERQPD256 */
        testfns0[i]  = copy_mvgatherqpd256;
        testname0[i] = "mvgatherqpd256";
        i++;
    }
#endif
#ifdef __AVX512F__
    {
        /* VMOVAPD512 */
        testfns0[i]  = copy_vmovapd512;
        testname0[i] = "vmovapd512";
        i++;
        /* VMOVUPD512 */
        testfns0[i]  = copy_vmovupd512;
        testname0[i] = "vmovupd512";
        i++;
        /* MVMOVAPD512 */
        testfns0[i]  = copy_mvmovapd512;
        testname0[i] = "mvmovapd512";
        i++;
        /* MVMOVUPD512 */
        testfns0[i]  = copy_mvmovupd512;
        testname0[i] = "mvmovupd512";
        i++;
        /* VMOVNTPD512 */
        testfns0[i]  = copy_vmovntpd512;
        testname0[i] = "vmovntpd512";
        i++;
        /* VMOVNTDQA512 */
        testfns0[i]  = copy_vmovntdqa512;
        testname0[i] = "vmovntdqa512";
        i++;
        /* VGSDPD512 */
        testfns0[i]  = copy_vGSdpd512;
        testname0[i] = "vGSdpd512";
        i++;
        /* MVGSDPD512 */
        testfns0[i]  = copy_mvGSdpd512;
        testname0[i] = "mvGSdpd512";
        i++;
        /* VGSQPD512 */
        testfns0[i]  = copy_vGSqpd512;
        testname0[i] = "vGSqpd512";
        i++;
        /* MVGSQPD512 */
        testfns0[i]  = copy_mvGSqpd512;
        testname0[i] = "mvGSqpd512";
        i++;
    }
#endif
    return i;
}

int setup_triad(void)
{
    int i = 0;
    {
        /* Reference */
        testfns1[i]  = triad_ref;
        testname1[i] = "Reference";
        i++;
        /* Official */
        testfns1[i]  = triad_official;
        testname1[i] = "Official";
        i++;
    }
#ifdef __SSE2__
    {
        /* MOVAPD128 */
        testfns1[i]  = triad_movapd128;
        testname1[i] = "movapd128";
        i++;
        /* MOVNTPD128 */
        testfns1[i]  = triad_movntpd128;
        testname1[i] = "movntpd128";
        i++;
    }
#endif
#ifdef __SSE4_1__
    {
        /* MOVNTDQ128 */
        testfns1[i]  = triad_movntdq128;
        testname1[i] = "movntdqa128";
        i++;
    }
#endif
#ifdef __FMA__
    {
        /* MOVAPD128+FMA */
        testfns1[i]  = triad_movapd128fma;
        testname1[i] = "movapd128fma";
        i++;
        /* MOVNTPD128+FMA */
        testfns1[i]  = triad_movntpd128fma;
        testname1[i] = "movntpd128fma";
        i++;
    }
#endif
#ifdef __AVX__
    {
        /* VMOVAPD256 */
        testfns1[i]  = triad_vmovapd256;
        testname1[i] = "vmovapd256";
        i++;
        /* VMOVNTPD256 */
        testfns1[i]  = triad_vmovntpd256;
        testname1[i] = "vmovntpd256";
        i++;
    }
#endif
#ifdef __AVX2__
    {
        /* VMOVNTDQA256 */
        testfns1[i]  = triad_vmovntdqa256;
        testname1[i] = "vmovntdqa256";
        i++;
    }
#endif
#ifdef __FMA__
    {
        /* VMOVAPD256+FMA */
        testfns1[i]  = triad_vmovapd256fma;
        testname1[i] = "vmovapd256fma";
        i++;
        /* VMOVNTPD256+FMA */
        testfns1[i]  = triad_vmovntpd256fma;
        testname1[i] = "vmovntpd256fma";
        i++;
        /* VMOVNTDQA256+FMA */
        testfns1[i]  = triad_vmovntdqa256fma;
        testname1[i] = "vmovntdqa256fma";
        i++;
    }
#endif
#ifdef __AVX512F__
    {
        /* VMOVAPD512 */
        testfns1[i]  = triad_vmovapd512;
        testname1[i] = "vmovapd512";
        i++;
        /* VMOVUPD512 */
        testfns1[i]  = triad_vmovupd512;
        testname1[i] = "vmovupd512";
        i++;
        /* MVMOVAPD512 */
        testfns1[i]  = triad_mvmovapd512;
        testname1[i] = "mvmovapd512";
        i++;
        /* MVMOVUPD512 */
        testfns1[i]  = triad_mvmovupd512;
        testname1[i] = "mvmovupd512";
        i++;
        /* VMOVNTPD512 */
        testfns1[i]  = triad_vmovntpd512;
        testname1[i] = "vmovntpd512";
        i++;
        /* VMOVNTDQA512 */
        testfns1[i]  = triad_vmovntdqa512;
        testname1[i] = "vmovntdqa512";
        i++;
        /* VGSDPD512 */
        testfns1[i]  = triad_vGSdpd512;
        testname1[i] = "vGSdpd512";
        i++;
        /* MVGSDPD512 */
        testfns1[i]  = triad_mvGSdpd512;
        testname1[i] = "mvGSdpd512";
        i++;
        /* VGSQPD512 */
        testfns1[i]  = triad_vGSqpd512;
        testname1[i] = "vGSqpd512";
        i++;
        /* MVGSQPD512 */
        testfns1[i]  = triad_mvGSqpd512;
        testname1[i] = "mvGSqpd512";
        i++;
    }
    {
        /* VMOVAPD512+FMA */
        testfns1[i]  = triad_vmovapd512fma;
        testname1[i] = "vmovapd512fma";
        i++;
        /* VMOVUPD512+FMA */
        testfns1[i]  = triad_vmovupd512fma;
        testname1[i] = "vmovupd512fma";
        i++;
        /* MVMOVAPD512+FMA */
        testfns1[i]  = triad_mvmovapd512fma;
        testname1[i] = "mvmovapd512fma";
        i++;
        /* MVMOVUPD512+FMA */
        testfns1[i]  = triad_mvmovupd512fma;
        testname1[i] = "mvmovupd512fma";
        i++;
        /* VMOVNTPD512+FMA */
        testfns1[i]  = triad_vmovntpd512fma;
        testname1[i] = "vmovntpd512fma";
        i++;
        /* VMOVNTDQA512+FMA */
        testfns1[i]  = triad_vmovntdqa512fma;
        testname1[i] = "vmovntdqa512fma";
        i++;
        /* VGSDPD512+FMA */
        testfns1[i]  = triad_vGSdpd512fma;
        testname1[i] = "vGSdpd512fma";
        i++;
        /* MVGSDPD512+FMA */
        testfns1[i]  = triad_mvGSdpd512fma;
        testname1[i] = "mvGSdpd512fma";
        i++;
        /* VGSQPD512+FMA */
        testfns1[i]  = triad_vGSqpd512fma;
        testname1[i] = "vGSqpd512fma";
        i++;
        /* MVGSQPD512+FMA */
        testfns1[i]  = triad_mvGSqpd512fma;
        testname1[i] = "mvGSqpd512fma";
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
