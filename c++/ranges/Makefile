CXX      = g++-11
CXXFLAGS = -g -O2 -std=c++20 -Wall -Wextra

TESTS = stride

all: $(TESTS)

%: %.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

