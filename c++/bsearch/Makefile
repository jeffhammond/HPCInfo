CXX      = icpc
CXXFLAGS = -O2 -std=c++14 -tbb

#CXX      = g++-7
#CXXFLAGS = -O2 -std=c++17

#CXX      = /opt/llvm/HEAD/bin/clang++
#CXXFLAGS = -g3 -O2 -std=c++17

# TBB crashes if you link it against libc++ if it was compiled against libstd++
#LIBCXX17 = -stdlib=libc++

TESTS = stl

all: $(TESTS)

stl: stl.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBCXX17) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

