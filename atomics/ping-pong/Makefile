OFLAGS   := -O3 -fopenmp -Wall

CC       := gcc-6
CXX      := g++-6

CFLAGS   := $(OFLAGS) -std=gnu11
CXXFLAGS := $(OFLAGS) -std=gnu++14

LIBS     = #-lpthread

LD       = $(CXX)
LDFLAGS  = $(OFLAGS)

TESTS = \
	c11-ping-pong   \
	cxx11-ping-pong \
	omp3-ping-pong  \
	omp4-ping-pong

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt

