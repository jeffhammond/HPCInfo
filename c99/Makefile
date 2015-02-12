CC      := gcc
CFLAGS   = -g -O2 -std=c99
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

TESTS = restrict warn-stack

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

