CXX      = g++
CXXFLAGS = -g -O2 -std=c++20

all: kahan.x

%.x: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -rf *.dSYM

