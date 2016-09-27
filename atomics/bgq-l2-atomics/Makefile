ifeq ($(TARGET),bgq)
OFLAGS   = -g3 -O3 -qsmp=omp
CC       = mpixlc_r 
CXX      = mpixlcxx_r
CFLAGS   = $(OFLAGS) -qlanglvl=extc99
CXXFLAGS = $(OFLAGS) -qlanglvl=extended0x
else
OFLAGS   := -O3 -Wall -fopenmp
CC       := gcc-5
CXX      := g++-5
CFLAGS   := $(OFLAGS) -std=gnu11
CXXFLAGS := $(OFLAGS) -std=gnu++11
endif

LIBS     = #-lpthread

LD       = $(CXX)
LDFLAGS  = $(OFLAGS)

TESTS = omp_atomic_counter bgq_l2_atomic_counter_sharing # c11_atomic_counter

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%: %.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt

