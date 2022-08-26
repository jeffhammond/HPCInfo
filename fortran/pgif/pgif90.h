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
