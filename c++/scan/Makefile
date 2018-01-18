CXX      = icpc
CXXFLAGS = -O2 -std=c++17 -tbb

#CXX      = g++-8
#CXXFLAGS = -O2 -std=c++17
#LIBS     =

#CXX      = /opt/llvm/HEAD/bin/clang++
#CXXFLAGS = -g3 -O2 -std=c++17

# TBB crashes if you link it against libc++ if it was compiled against libstd++
#LIBCXX17 = -stdlib=libc++
#LIBTBB   = -ltbb

TESTS = scan tbb

all: $(TESTS)

scan: scan.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBCXX17) -o $@

tbb: tbb.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBTBB) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

