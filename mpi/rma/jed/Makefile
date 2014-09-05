CC      = mpicc
CFLAGS  = -g -O2 -std=c99 -Wall

LD      = $(CC)
LDFLAGS = $(CFLAGS) -lquadmath

all: math.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -rf *.dSYM
