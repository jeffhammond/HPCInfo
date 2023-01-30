CC      = mpicc
CFLAGS	= -g -O2 -Wall

all: perf.x

%.x: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
