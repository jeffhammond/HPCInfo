CC       = mpicc
CFLAGS   = -g -O2 -Wall -Wextra
LIBS     = -lpthread

TESTS = pthread_affinity

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM


