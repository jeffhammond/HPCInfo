#CC = gcc-14
#CXX = g++-14
CC = clang
CXX = clang++
FC = gfortran-14

CFIFLAGS = -I.

CFLAGS = -g -std=c11 $(CFIFLAGS)
CXXFLAGS = -g -std=c++2a $(CFIFLAGS)
FCFLAGS = -g -std=f2018

all: class.x array.x

%.x: %.F90 foo.o mmdspan.o
	$(FC) $(FCFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-rm -fr *.dSYM
	-rm -f *.x
	-rm -f *.mod
	-rm -f *.o
	-rm -f a.out

