CXX       = mpicxx
CXXFLAGS  = -g -O2 -Wall -std=c++14
LIBS      =

all: join.x join2.x

%.x: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -fr *.x.dSYM
