OPT	 = -O3

CXX = if [ x"$(shell which icpc)" != x"" ] ; then       \
        echo "icpc" ;                                   \
      elif [ x"$(shell which g++-5)" != x"" ] ; then    \
        echo "g++-5" ;                                  \
      elif [ x"$(shell which clang++)" != x"" ] ; then  \
        echo "clang++" ;                                \
      else                                              \
        echo "false" ;                                  \
      endif

# Assume these flags work with all compilers, which should
# at least be true for GCC, Clang and Intel.
CXXFLAGS = $(OPT) -std=c++11

ifeq ($(CXX),icpc)
    # These are for recent releases of the Intel compiler
    INCLUDE  = -tbb
    LIBS     = -tbb
else
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

