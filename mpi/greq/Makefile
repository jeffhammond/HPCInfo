CC      = mpicc
COPT    = -g -O2 -Wall

all: nb_reduce_thread.x

%.x: %.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -rf *.dSYM
