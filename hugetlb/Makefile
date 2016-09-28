OFLAGS   := -O3 -fopenmp -Wall -fPIC

# 2MB and 1GB mmap flags require Linux 3.8 or later.
#OFLAGS += -DLINUX_MAJOR=3 -DLINUX_MINOR=8
#UNAME := $(shell uname -r)
#OFLAGS += -DLINUX_MAJOR=${LINUX_MAJOR} -DLINUX_MINOR=${LINUX_MINOR}

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

TESTS = huge-alloc.o

all: $(TESTS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.x: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt

realclean: clean
	-rm -f a.out
	-rm -f *.x

