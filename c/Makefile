CC      := gcc-12
CFLAGS   = -g -O3 -Wall -std=c99
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

TESTS = warn-stack overflow

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt

