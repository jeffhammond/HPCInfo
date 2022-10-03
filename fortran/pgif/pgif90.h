/*
 *     Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// from /opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/include_acc/nvhpc_cuda_runtime.h

#define MAXDIMS 7

struct F90_DescDim_la {
  long long lbound;
  long long extent;
  long long sstride;
  long long soffset;
  long long lstride;
  long long ubound;
};

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
