CXX		= g++-8
CXXFLAGS	= -fopenmp

all: simple wrap

%: %.cc
	${CXX} ${CXXFLAGS} $< -o $@

clean:
	-rm -f *.o
	-rm -f simple
	-rm -f wrap
	-rm -rf *.dSYM
