CXX      = g++-9
CXXFLAGS = -O2 -std=c++17

TESTS = pmr.x

all: $(TESTS)

pmr.x: pmr.cc
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

