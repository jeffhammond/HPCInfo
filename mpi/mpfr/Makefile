CC      = mpicc
CFLAGS  = -g -O2 -Wall -std=c99

all: mpfr-type.x

%.x: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -rf *.dSYM
