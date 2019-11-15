CXX      = g++-9
CXXFLAGS = -g -O2 -std=c++17 #-Wall #-Wextra -Wno-unused-parameter

# RANGES
CXXFLAGS += -I$(HOME)/Work/Languages/range-v3/include

# TBB
CXXFLAGS += -I/usr/local/Cellar/tbb/2019_U9/include
LIBS     += -L/usr/local/Cellar/tbb/2019_U9/lib -ltbb

# SYCL
CXXFLAGS += -I$(HOME)/Work/OpenCL/triSYCL/include
CXXFLAGS += -I/usr/local/Cellar/boost/1.71.0/include

TESTS = stl_for.x stl_for_each.x stl_for_each_pstl.x stl_for_each_tbb.x sycl_for.x usm_sycl_for.x

all: $(TESTS)

%.x: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

