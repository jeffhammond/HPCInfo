CXX        = mpicxx
CXXFLAGS   = -g -O2 -Wall -std=c++11

all: test_large_count_type

test_large_count_type: test_large_count_type.cpp large_count_type.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm -f test_large_count_type.o

realclean: clean
	-rm -f test_large_count_type
