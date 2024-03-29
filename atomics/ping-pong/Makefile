ifndef COMPILER
    COMPILER=gcc
endif

ifeq ($(COMPILER),gcc)
CC       := gcc
CXX      := g++
ASMFLAGS  = -S -fverbose-asm
else ifeq ($(COMPILER),clang)
CC       := /usr/local/Cellar/llvm/5.0.0/bin/clang
CXX      := /usr/local/Cellar/llvm/5.0.0/bin/clang++
OMPFLAGS  = -I/usr/local/Cellar/libiomp/20150701/include/libiomp
OMPLIBS   = -L/usr/local/Cellar/libiomp/20150701/lib -liomp5
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

OFLAGS += -march=armv8.2-a

ifeq ($(model),sc)
    OFLAGS   += -DSEQUENTIAL_CONSISTENCY
    BINARY    = sc.$(COMPILER)
else
    BINARY    = ar.$(COMPILER)
endif
CFLAGS   += $(OFLAGS)
CXXFLAGS += $(OFLAGS)

LIBS     = $(OMPLIBS)

LD       = $(CXX)
LDFLAGS  = $(OFLAGS)

TESTS = cxx11-ping-pong.$(BINARY) \
	c99-omp-ping-pong.$(BINARY) \
	c11-ping-pong.$(BINARY) \
	gcc-ping-pong.$(BINARY) \
	cxx11-dekker.$(BINARY) \
	cxx11-dekker2.$(BINARY) \
	c99-omp-dekker.$(BINARY)

ASM =   cxx11-ping-pong.$(BINARY).s \
	c99-omp-ping-pong.$(BINARY).s \
	c11-ping-pong.$(BINARY).s \
	gcc-ping-pong.$(BINARY).s \
	cxx11-dekker.$(BINARY).s \
	cxx11-dekker2.$(BINARY).s \
	c99-omp-dekker.$(BINARY).s

all: $(TESTS)

asm: $(ASM)

# binaries

%.$(BINARY): %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%.$(BINARY): %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

# assembly

%.$(BINARY).s: %.c
	$(CC) $(CFLAGS) $(ASMFLAGS) $< -o $@

%.$(BINARY).s: %.cc
	$(CXX) $(CXXFLAGS) $(ASMFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt

realclean: clean
	-rm -f a.out
	-rm -f *.gcc
	-rm -f *.clang
	-rm -f *.icc
	-rm -f *.intel


