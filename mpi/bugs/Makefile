CC      = mpicc
CFLAGS  = -g -O2 -Wall -std=c99

all: iprobe-overflow.x sendrecv_replace.x max-irecv.x detect-mismatch.x

%.x: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -rf *.dSYM
