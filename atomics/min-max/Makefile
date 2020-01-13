ifndef COMPILER
    COMPILER=gcc
endif

ifeq ($(COMPILER),gcc)
CC       := gcc-9
CXX      := g++-9
FC	 := gfortran-9
ASMFLAGS  = -S -fverbose-asm
else ifeq ($(COMPILER),clang)
LLVM_VERSION=9.0.0_1
KMP_VERSION=9.0.0
CC       := /usr/local/Cellar/llvm/$(LLVM_VERSION)/bin/clang
CXX      := /usr/local/Cellar/llvm/$(LLVM_VERSION)/bin/clang++
OMPFLAGS  = -I/usr/local/Cellar/libiomp/$(KMP_VERSION)/include/libiomp
OMPLIBS   = -L/usr/local/Cellar/libiomp/$(KMP_VERSION)/lib -liomp5
ASMFLAGS  = -S -fverbose-asm
# Clang tries to link libomp, so we symlink this to libiomp5
else ifeq ($(COMPILER),intel)
CC       := icc
CXX      := icpc
FC	 := ifort
ASMFLAGS  = -S -fverbose-asm -fcode-asm -fasm-blocks
else ifeq ($(COMPILER),pgi)
CC       := pgcc
CXX      := pgc++
FC	 := pgfortran
ASMFLAGS  =
else
CC       := false
CXX      := false
FC	 := false
endif

ifeq ($(COMPILER),pgi)
    OFLAGS   := -O3 -mp
    CFLAGS   := $(OFLAGS) -c11
    CXXFLAGS := $(OFLAGS) --c++11
    FCFLAGS	 := $(OFLAGS) -Mpreprocess -Mfreeform
else
    ifeq ($(COMPILER),intel)
        OFLAGS   := -O3 -qopenmp
        CFLAGS   := $(OFLAGS) -Wall -fPIC -std=gnu11
        CXXFLAGS := $(OFLAGS) -Wall -fPIC -std=gnu++11
        FCFLAGS	 := $(OFLAGS) -warn all -free -std08 -fpp
    else
        OFLAGS   := -O3 -fopenmp -Wall -fPIC
        CFLAGS   := $(OFLAGS) $(OMPFLAGS) -std=gnu11 -fno-strict-aliasing
        CXXFLAGS := $(OFLAGS) $(OMPFLAGS) -std=gnu++17 -fno-strict-aliasing
        FCFLAGS	 := $(OFLAGS) -ffree-form -std=f2008
    endif
endif

# Apple debug
#OFLAGS += -g3 -Wl,-pie

#OFLAGS   += -DSEQUENTIAL_CONSISTENCY

LIBS     = $(OMPLIBS)

TESTS = c99-omp-min-max.$(COMPILER) \
	gcc-min-max.$(COMPILER) \
	f90-min-max.$(COMPILER)

ASM =   c99-omp-min-max.$(COMPILER).s \
	gcc-min-max.$(COMPILER).s \
	f90-min-max.$(COMPILER).s


all: $(TESTS)

asm: $(ASM)

# binaries

%.$(COMPILER): %.F
	$(FC) $(FCFLAGS) $< $(LIBS) -o $@

%.$(COMPILER): %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%.$(COMPILER): %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

# assembly

%.$(COMPILER).s: %.F
	$(FC) $(FCFLAGS) $(ASMFLAGS) $< -o $@

%.$(COMPILER).s: %.c
	$(CC) $(CFLAGS) $(ASMFLAGS) $< -o $@

%.$(COMPILER).s: %.cc
	$(CXX) $(CXXFLAGS) $(ASMFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt
	-rm -f a.out
	-rm -f *.gcc
	-rm -f *.clang
	-rm -f *.icc
	-rm -f *.intel


