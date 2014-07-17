CC       = clang-3.5
CFLAGS   = -g -O2 -std=c11
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

all: atomics.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x

distclean: realclean

