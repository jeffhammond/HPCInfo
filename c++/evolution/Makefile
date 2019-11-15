CXX      = g++-9
CXXFLAGS = -g -O2 -std=c++17 -Wall -Wextra

# RANGES
CXXFLAGS += -I$(HOME)/Work/Languages/range-v3/include

# TBB
CXXFLAGS += -I/usr/local/Cellar/tbb/2019_U9/include
LIBS     += -L/usr/local/Cellar/tbb/2019_U9/lib -ltbb

TESTS = stl_for.x stl_for_each.x stl_for_each_pstl.x stl_for_each_tbb.x

all: $(TESTS)

%.x: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

