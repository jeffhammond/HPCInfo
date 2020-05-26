CXX       = mpicxx
CXXFLAGS  = -g -O2 -Wall -std=c++14
LIBS      =

all: join.x

%.x: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -fr *.x.dSYM
