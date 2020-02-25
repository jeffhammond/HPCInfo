CC       = mpicc
CFLAGS   = -g -O2
LIBS     =

TESTS = interprocess_mutex interprocess_mutex_mpi

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM


