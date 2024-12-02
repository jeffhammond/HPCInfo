CC = gcc-14
CXX = g++-14
#LLVM_PATH = /opt/homebrew/opt/llvm/bin/
#CC = $(LLVM_PATH)clang
#CXX = $(LLVM_PATH)clang++
FC = gfortran-14

CFIFLAGS = -I.

CFLAGS = -g -std=c11 $(CFIFLAGS)
CXXFLAGS = -g -std=c++2a $(CFIFLAGS)
FCFLAGS = -g -std=f2018

all: class.x array.x

%.x: %.F90 foo.c.o mmdspan.cc.o
	$(FC) $(FCFLAGS) $^ -o $@

%.x: %.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cc.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-rm -fr *.dSYM
	-rm -f *.x
	-rm -f *.mod
	-rm -f *.o
	-rm -f a.out

