CXX      = g++-10
CXXFLAGS = -g -O2 -std=c++2a

all: hardware

%: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f hardware
	-rm -rf *.dSYM

