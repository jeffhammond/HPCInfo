CC      := gcc
CFLAGS   = -g -O2 -std=c99
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

all: restrict

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -rf *.dSYM

