CC      := icc
CFLAGS   = -g -O3 -Wall -std=c99
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

TESTS = restrict warn-stack array-restrict

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

