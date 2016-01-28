OFLAGS   := -O3 -Wall

CC       := gcc-5
CXX      := g++-5

CFLAGS   := $(OFLAGS) -std=c11
CXXFLAGS := $(OFLAGS) -std=c++11

LIBS      =

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

