# NaN behaviour of + * min max across C, x87/SSE/AVX/AVX-512, NEON/SVE, CUDA, OpenCL.
#
#   make              build with everything that is detected
#   make run          build and run
#   make info         show which backends were enabled
#
# Overrides:
#   ENABLE_CUDA=0|1  ENABLE_OPENCL=0|1  ENABLE_SVE=0|1  ENABLE_FMAXIMUM=0|1
#   CUDA_ARCH=sm_80 (default: native, needs CUDA >= 11.5)
#   OPENCL_CPPFLAGS=-I/usr/local/cuda/include   OPENCL_LIBS=-L/path -lOpenCL
#   CPPFLAGS=-DNT_NO_AVX512   (skip AVX-512 code at compile time)

CC       ?= cc
NVCC     ?= nvcc
CFLAGS   ?= -O2 -g
CPPFLAGS += -D_GNU_SOURCE

WARN     = -Wall -Wextra
# the whole point is to observe NaNs, so never let the compiler assume they cannot occur
NOFAST   = -fno-fast-math -fno-finite-math-only
PROBE_FLAGS := -std=gnu11 $(CPPFLAGS)

ALL_CFLAGS = -std=gnu11 $(CFLAGS) $(WARN) $(NOFAST) $(CPPFLAGS)

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ENABLE_CUDA     ?= auto
ENABLE_OPENCL   ?= auto
ENABLE_SVE      ?= auto
ENABLE_FMAXIMUM ?= auto

OBJS = nan_minmax.o nan_x86.o nan_arm.o
LIBS = -lm
LINK = $(CC)

# ---- C23 fmaximum / fminimum (glibc 2.41+, recent libms) ----
ifeq ($(ENABLE_FMAXIMUM),auto)
 ifeq ($(shell $(CC) $(PROBE_FLAGS) probes/fmaximum.c -o /dev/null -lm >/dev/null 2>&1 && echo 1),1)
  ENABLE_FMAXIMUM := 1
 else
  ENABLE_FMAXIMUM := 0
 endif
endif
ifeq ($(ENABLE_FMAXIMUM),1)
 CPPFLAGS += -DHAVE_FMAXIMUM
endif

# ---- Arm SVE (only meaningful on aarch64) ----
SVE_FLAGS = -march=armv8-a+sve
ifeq ($(ENABLE_SVE),auto)
 ENABLE_SVE := 0
 ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
  ifeq ($(shell $(CC) $(PROBE_FLAGS) $(SVE_FLAGS) -c probes/sve.c -o /dev/null >/dev/null 2>&1 && echo 1),1)
   ENABLE_SVE := 1
  endif
 endif
endif
ifeq ($(ENABLE_SVE),1)
 OBJS     += nan_sve.o
 CPPFLAGS += -DNT_SVE
endif

# ---- OpenCL ----
ifeq ($(UNAME_S),Darwin)
 OPENCL_LIBS ?= -framework OpenCL
else
 OPENCL_LIBS ?= -lOpenCL
endif
CL_CPPFLAGS = -DCL_TARGET_OPENCL_VERSION=120 -DCL_SILENCE_DEPRECATION $(OPENCL_CPPFLAGS)
ifeq ($(ENABLE_OPENCL),auto)
 ifeq ($(shell $(CC) $(PROBE_FLAGS) $(CL_CPPFLAGS) probes/opencl.c -o /dev/null $(OPENCL_LIBS) >/dev/null 2>&1 && echo 1),1)
  ENABLE_OPENCL := 1
 else
  ENABLE_OPENCL := 0
 endif
endif
ifeq ($(ENABLE_OPENCL),1)
 OBJS     += nan_opencl.o
 CPPFLAGS += -DNT_OPENCL
 LIBS     += $(OPENCL_LIBS)
endif

# ---- CUDA ----
CUDA_ARCH ?= native
NVCCFLAGS ?= -O2
ifeq ($(ENABLE_CUDA),auto)
 ifneq ($(shell command -v $(NVCC) 2>/dev/null),)
  ENABLE_CUDA := 1
 else
  ENABLE_CUDA := 0
 endif
endif
ifeq ($(ENABLE_CUDA),1)
 OBJS     += nan_cuda.o
 CPPFLAGS += -DNT_CUDA
 LINK      = $(NVCC)    # nvcc links the CUDA runtime
endif

# ---- rules ----
all: nan_minmax

nan_minmax: $(OBJS)
	$(LINK) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

$(OBJS): nan_common.h

%.o: %.c
	$(CC) $(ALL_CFLAGS) -c $< -o $@

nan_sve.o: nan_sve.c
	$(CC) $(ALL_CFLAGS) $(SVE_FLAGS) -c $< -o $@

nan_opencl.o: nan_opencl.c
	$(CC) $(ALL_CFLAGS) $(CL_CPPFLAGS) -c $< -o $@

nan_cuda.o: nan_cuda.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -DNT_CUDA -c $< -o $@

run: nan_minmax
	./nan_minmax

info:
	@echo "machine  : $(UNAME_S) $(UNAME_M)"
	@echo "CC       : $(CC)"
	@echo "fmaximum : $(ENABLE_FMAXIMUM)"
	@echo "SVE      : $(ENABLE_SVE)"
	@echo "OpenCL   : $(ENABLE_OPENCL)"
	@echo "CUDA     : $(ENABLE_CUDA)  (arch $(CUDA_ARCH))"
	@echo "objects  : $(OBJS)"

clean:
	rm -f *.o nan_minmax

.PHONY: all run info clean
