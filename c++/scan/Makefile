#CXX      = icpc
#CXXFLAGS = -O2 -std=c++17 -gxx-name=g++-7
#LIBS     = -tbb

#CXX      = g++-8
#CXXFLAGS = -O2 -std=c++17
#LIBS     =

CXX      = /opt/llvm/HEAD/bin/clang++
CXXFLAGS = -O2 -std=c++17 -stdlib=libc++
LIBS     =

TESTS = scan

all: $(TESTS)

%: %.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

