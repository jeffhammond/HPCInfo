#!/bin/bash +x

OPT	 = -O3

# This is only here to test the logic
#HAVE_GCC5  = $(shell which g++-5)
#ifeq ($(HAVE_GCC5),) 
#  CXX=clang++
#else
#  CXX=g++-5
#endif

HAVE_INTEL = $(shell which icpc)
ifeq ($(HAVE_INTEL),) 
  CXX=clang++
else
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

