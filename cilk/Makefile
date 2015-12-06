CC       = icc
CFLAGS   = -g -O2 -std=c99
LD       = $(CC)
LDFLAGS  = $(CFLAGS)
LIBS     =

all: for1.x spawn1.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x

