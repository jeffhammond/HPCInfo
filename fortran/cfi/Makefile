#CC = gcc-12
#CXX = g++-12
#LLVM_PATH = /opt/homebrew/opt/llvm/bin/
LLVM_VERSION=
CC = $(LLVM_PATH)clang$(LLVM_VERSION)
CXX = $(LLVM_PATH)clang++$(LLVM_VERSION)
FC = gfortran

# git clone https://github.com/kokkos/mdspan.git kokkos-mdspan
MDSPAN_FLAGS = -I./kokkos-mdspan/include

CFIFLAGS = -I.

CFLAGS = -g -std=c11 $(CFIFLAGS)
CXXFLAGS = -g -std=c++20 $(CFIFLAGS) $(MDSPAN_FLAGS)
FCFLAGS = -g -std=f2018
#LIBCXX = -lstdc++ # Linux
LIBCXX = -lc++ # MacOS

all: class.x array.x

%.x: %.F90 foo.c.o mmdspan.cc.o
	$(FC) $^ $(FCFLAGS) $(LIBCXX) -o $@

%.x: %.cc
	$(CXX) $^ $(CXXFLAGS) -o $@

%.c.o: %.c
	$(CC) -c $< $(CFLAGS) -o $@

%.cc.o: %.cc
	$(CXX) -c $< $(CXXFLAGS) -o $@

clean:
	-rm -fr *.dSYM
	-rm -f *.x
	-rm -f *.mod
	-rm -f *.o
	-rm -f a.out

