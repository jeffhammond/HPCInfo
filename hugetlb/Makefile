OFLAGS   := -O3 -fopenmp -Wall -fPIC

# Apple debug 
# OFLAGS += -g3 -Wl,-pie

ifndef COMPILER
    COMPILER=gcc
endif

ifeq ($(COMPILER),gcc)
CC       := gcc
CXX      := g++
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

TESTS =

all: $(TESTS)

# binaries

%.x: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%.x: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt

realclean: clean
	-rm -f a.out
	-rm -f *.x

