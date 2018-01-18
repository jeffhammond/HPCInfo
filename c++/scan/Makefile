CXX      = icpc
CXXFLAGS = -O2 -std=c++14
LIBS     = -tbb

TESTS = scan

all: $(TESTS)

%: %.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

