CC       = mpicc
CFLAGS   = -g -O2 -Wall -Wextra
LIBS     = -lpthread

FC	 = mpifort
FCFLAGS	 = -g -O2

TESTS = pthread_affinity vv

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%: %.F90 v.o
	$(FC) -DTEST $(FCFLAGS) $^ $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM


