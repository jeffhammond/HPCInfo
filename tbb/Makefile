OPT	 = -O3

CXX = if test -e icpc ; then echo "icpc" ; else echo "clang++" ; fi

ifeq ($(CXX),icpc)
    CXX      = icpc
    CXXFLAGS = $(OPT) -std=c++11
    # These are for recent releases of the Intel compiler
    INCLUDE  = -tbb
    LIBS     = -tbb
else
    CXX	 = clang++
    CXXFLAGS = $(OPT) -std=c++11
    # These are for Homebrew OSS TBB
    INCLUDE  = -I/usr/local/include
    LIBS     = -L/usr/local/lib
endif

TESTS = hello.x

all: $(TESTS)

%.x: %.o
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)

