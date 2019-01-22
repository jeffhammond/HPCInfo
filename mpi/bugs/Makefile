CC      = mpicc
CFLAGS  = -g -O2 -Wall -std=c99

all: iprobe-overflow.x

%.x: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x
