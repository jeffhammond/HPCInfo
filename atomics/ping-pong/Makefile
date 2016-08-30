OFLAGS   := -O3 -fopenmp -Wall -g3 -Wl,-pie

#OFLAGS   += -DSEQUENTIAL_CONSISTENCY

#CC       := gcc-6
#CXX      := g++-6

CC        :=clang-3.8
CXX       :=clang++-3.8
# Mac clang-3.x only
OMPFLAGS = -I/usr/local/Cellar/libiomp/20150701/include/libiomp
OMPLIBS  = -L/usr/local/Cellar/libiomp/20150701/lib -liomp5 -L.


CFLAGS   := $(OFLAGS) $(OMPFLAGS) -std=gnu11
CXXFLAGS := $(OFLAGS) $(OMPFLAGS) -std=gnu++14

LIBS     = $(OMPLIBS)

LD       = $(CXX)
LDFLAGS  = $(OFLAGS)

TESTS = \
	c11-ping-pong   \
	cxx11-ping-pong \
	c99-omp-ping-pong

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

