CXX=clang++
CXXFLAGS=-std=c++20 -Wall -pedantic -O3

all: bogosort.x

%.x: %.cc
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
