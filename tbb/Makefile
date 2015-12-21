OPT	 = -O3

# Unused
#ifeq ($(shell uname),Darwin) 
#    HAVE_MAC = y
#else
#    HAVE_MAC = n
#endif

HAVE_INTEL = $(shell which icpc)
HAVE_CLANG = $(shell which clang++)
HAVE_GCC5  = $(shell which g++-5)
ifeq ($(HAVE_INTEL),) 
  ifeq ($(HAVE_CLANG),) 
    ifeq ($(HAVE_GCC5),) 
      # last choice - should be fine on Linux but bad on Mac
      CXX=g++
    else
      # second-to-last choice - here because of Mac
      CXX=g++-5
    endif
  else
    # second choice
    CXX=clang++
  endif
else
  # first choice
  CXX=icpc
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

