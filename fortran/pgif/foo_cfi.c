#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// from /opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/include_acc/nvhpc_cuda_runtime.h

#define MAXDIMS 7

struct F90_DescDim {
  int lbound;
  int extent;
  int sstride;
  int soffset;
  int lstride;
  int ubound;
};

struct F90_DescDim_la {
  long long lbound;
  long long extent;
  long long sstride;
  long long soffset;
  long long lstride;
  long long ubound;
};

typedef struct F90_Desc {
  int tag;
  int rank;
  int kind;
  int len;
  int flags;
  int lsize;
  int gsize;
  int lbase;
  int *gbase;
  int *unused;
  struct F90_DescDim dim[MAXDIMS];
} F90_Desc;

typedef struct F90_Desc_la {
  long long tag;
  long long rank;
  long long kind;
  long long len;
  long long flags;
  long long lsize;
  long long gsize;
  long long lbase;
  long long *gbase;
  long long *unused;
  struct F90_DescDim_la dim[MAXDIMS];
} F90_Desc_la;

void foo(int * buffer, int * m, int * n, int * o)
{
    printf("FOO buffer = %p\n", buffer);
    printf("FOO m,n,o = %d,%d,%d\n", *m, *n, *o);
}

void print_flags(long long flags)
{
    bool * TEMPLATE           = flags & 0x00010000;
    bool * OFF_TEMPLATE       = flags & 0x00080000;
    bool * SECTZBASE          = flags & 0x00400000;
    bool * BOGUSBOUNDS        = flags & 0x00800000;
    bool * NOT_COPIED         = flags & 0x01000000;
    bool * NOREINDEX          = flags & 0x02000000;
    bool * SEQUENTIAL_SECTION = flags & 0x20000000;

    printf("TEMPLATE           = %s\n", TEMPLATE           ? "true" : "false");
    printf("OFF_TEMPLATE       = %s\n", OFF_TEMPLATE       ? "true" : "false");
    printf("SECTZBASE          = %s\n", SECTZBASE          ? "true" : "false");
    printf("BOGUSBOUNDS        = %s\n", BOGUSBOUNDS        ? "true" : "false");
    printf("NOT_COPIED         = %s\n", NOT_COPIED         ? "true" : "false");
    printf("NOREINDEX          = %s\n", NOREINDEX          ? "true" : "false");
    printf("SEQUENTIAL_SECTION = %s\n", SEQUENTIAL_SECTION ? "true" : "false");
}

void bar(int * buffer, int * m, int * n, int * o, F90_Desc_la * d)
{
    printf("BAR buffer = %p\n", buffer);
    printf("BAR m,n,o = %d,%d,%d\n", *m, *n, *o);
    printf("BAR F90_Desc = %p\n", d);
    printf("BAR F90_Desc->tag   = %lld\n", d->tag  );
    printf("BAR F90_Desc->rank  = %lld\n", d->rank );
    printf("BAR F90_Desc->kind  = %lld\n", d->kind );
    printf("BAR F90_Desc->len   = %lld\n", d->len  );
    printf("BAR F90_Desc->flags = %lld\n", d->flags);
    print_flags(d->flags);
    printf("BAR F90_Desc->lsize = %lld\n", d->lsize);
    printf("BAR F90_Desc->gsize = %lld\n", d->gsize);
#if 1
    printf("BAR F90_Desc->lbase = %lld\n", d->lbase);
    printf("BAR F90_Desc->gbase = %p\n",   d->gbase);
    for (int i=0; i<d->rank; i++) {
        printf("BAR F90_Desc->dim.lbound  = %lld\n", d->dim[i].lbound );
        printf("BAR F90_Desc->dim.extent  = %lld\n", d->dim[i].extent );
        printf("BAR F90_Desc->dim.sstride = %lld\n", d->dim[i].sstride);
        printf("BAR F90_Desc->dim.soffset = %lld\n", d->dim[i].soffset);
        printf("BAR F90_Desc->dim.lstride = %lld\n", d->dim[i].lstride);
        printf("BAR F90_Desc->dim.ubound  = %lld\n", d->dim[i].ubound );
    }
#endif
}
