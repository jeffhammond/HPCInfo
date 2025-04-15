#CC = gcc-12
#CXX = g++-12
#LLVM_PATH = /opt/homebrew/opt/llvm/bin/
LLVM_VERSION=
CXX = $(LLVM_PATH)clang++$(LLVM_VERSION)

# git clone https://github.com/kokkos/mdspan.git kokkos-mdspan
MDSPAN_FLAGS = -I./kokkos-mdspan/include

CXXFLAGS = -g -std=c++20 $(MDSPAN_FLAGS)

all: mdspan.x

%.x: %.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	-rm -fr *.dSYM
	-rm -f *.x
	-rm -f *.mod
	-rm -f *.o
	-rm -f a.out

