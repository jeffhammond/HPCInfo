CC      = mpicc
COPT    = -g -O2 -Wall -std=c11

LD      = $(CC)
CFLAGS  = $(COPT)

LDFLAGS = $(COPT)
LIBS    = -lm

all: c-logical-reduction.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
