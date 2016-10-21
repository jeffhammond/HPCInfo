OFLAGS   := -O3 -fopenmp -Wall -fPIC

# Apple debug 
 OFLAGS += -g3 -Wl,-pie

#OFLAGS   += -DSEQUENTIAL_CONSISTENCY

ifndef COMPILER
    COMPILER=gcc
endif

ifeq ($(COMPILER),gcc)
CC       := gcc-6
CXX      := g++-6
ASMFLAGS  = -S -fverbose-asm
else ifeq ($(COMPILER),clang)
CC       := clang-3.8
CXX      := clang++-3.8
OMPFLAGS  = -I/usr/local/Cellar/libiomp/20150701/include/libiomp
OMPLIBS   = -L/usr/local/Cellar/libiomp/20150701/lib -liomp5
ASMFLAGS  = -S -fverbose-asm
# Clang tries to link libomp, so we symlink this to libiomp5
else ifeq ($(COMPILER),intel)
CC       := icc
CXX      := icpc
ASMFLAGS  = -S -fverbose-asm -fcode-asm -fasm-blocks
else
CC       := false
CXX      := false
endif

CFLAGS   := $(OFLAGS) $(OMPFLAGS) -std=gnu11
CXXFLAGS := $(OFLAGS) $(OMPFLAGS) -std=gnu++11

LIBS     = $(OMPLIBS)

LD       = $(CXX)
LDFLAGS  = $(OFLAGS)

TESTS = cxx11-ping-pong.$(COMPILER) \
	c99-omp-ping-pong.$(COMPILER) \
	c11-ping-pong.$(COMPILER) \
	gcc-ping-pong.$(COMPILER) \
	cxx11-dekker.$(COMPILER) \
	cxx11-dekker2.$(COMPILER) \
	c99-omp-dekker.$(COMPILER)

ASM =   cxx11-ping-pong.$(COMPILER).s \
	c99-omp-ping-pong.$(COMPILER).s \
	c11-ping-pong.$(COMPILER).s \
	gcc-ping-pong.$(COMPILER).s \
	cxx11-dekker.$(COMPILER).s \
	cxx11-dekker2.$(COMPILER).s \
	c99-omp-dekker.$(COMPILER).s

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

realclean: clean
	-rm -f a.out
	-rm -f *.gcc
	-rm -f *.clang
	-rm -f *.icc
	-rm -f *.intel


