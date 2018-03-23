CXX      = g++-7
CXXFLAGS = -g -O2 -std=c++14

CXXFLAGS += -I$(HOME)/Work/Languages/ranges-impl/include

TESTS = ranges

all: $(TESTS)

%: %.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

