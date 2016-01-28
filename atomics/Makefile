OFLAGS   := -O3 -Wall

CC       := gcc
CXX      := g++

CFLAGS   := $(OFLAGS) -std=gnu11
CXXFLAGS := $(OFLAGS) -std=gnu++11

LIBS      = -lpthread

LD       = $(CXX)
LDFLAGS  = $(OFLAGS)

TESTS = c11_atomic_counter

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

