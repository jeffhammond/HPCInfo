ifndef COMPILER
    COMPILER=gcc
endif

ifeq ($(COMPILER),gcc)
CC       := gcc-9
CXX      := g++-9
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
ASMFLAGS  = -S -fverbose-asm -fcode-asm -fasm-blocks
else ifeq ($(COMPILER),pgi)
CC       := pgcc
CXX      := pgc++
ASMFLAGS  =
else
CC       := false
CXX      := false
endif

ifeq ($(COMPILER),pgi)
OFLAGS   := -O3 -mp
CFLAGS   := $(OFLAGS) -c11
CXXFLAGS := $(OFLAGS) --c++11
else
OFLAGS   := -O3 -fopenmp -Wall -fPIC
CFLAGS   := $(OFLAGS) $(OMPFLAGS) -std=gnu11
CXXFLAGS := $(OFLAGS) $(OMPFLAGS) -std=gnu++11
endif

# Apple debug
#OFLAGS += -g3 -Wl,-pie

#OFLAGS   += -DSEQUENTIAL_CONSISTENCY

LIBS     = $(OMPLIBS)

LD       = $(CXX)
LDFLAGS  = $(OFLAGS)

TESTS = cxx11-min-max.$(COMPILER) \
	c99-omp-min-max.$(COMPILER) \
	c11-min-max.$(COMPILER) \
	gcc-min-max.$(COMPILER)

ASM =   cxx11-min-max.$(COMPILER).s \
	c99-omp-min-max.$(COMPILER).s \
	c11-min-max.$(COMPILER).s \
	gcc-min-max.$(COMPILER).s

all: $(TESTS)

asm: $(ASM)

# binaries

%.$(COMPILER): %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%.$(COMPILER): %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

# assembly

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


