CXX      = g++-9
CXXFLAGS = -g -O2 -std=c++17 -Wall -Wextra

CXXFLAGS += -I$(HOME)/Work/Languages/range-v3/include

TESTS = stl_for.x stl_for_each.x stl_for_each_pstl.x

all: $(TESTS)

%.x: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

