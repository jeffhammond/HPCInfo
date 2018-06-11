CXX      = clang++
CXXFLAGS = -g -O2 -std=c++11 -Wall -Wextra

CXXFLAGS += -I$(HOME)/Work/Languages/range-v3/include

TESTS = ranges

all: $(TESTS)

%: %.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

