CC       = gcc
CFLAGS   = -g -O2 -std=c99
LIBS     =

TESTS = gtod

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm  -f *.o
	-rm -rf *.dSYM
	-rm  -f $(TESTS)
