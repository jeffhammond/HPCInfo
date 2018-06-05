CXX		= g++-8
CXXFLAGS	= -fopenmp

all: simple

simple: simple.cc
	${CXX} ${CXXFLAGS} $< -o $@

clean:
	-rm -f *.o
	-rm -f simple
	-rm -rf *.dSYM
