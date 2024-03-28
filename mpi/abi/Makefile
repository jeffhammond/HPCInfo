CC      = mpicc
COPT    = -g -O2 -Wall -std=c11

FC      = mpifort
FOPT    = -g -O2 -Wall -std=f2018

LD      = mpifort

LDFLAGS =
LIBS    = -lm

all: c-logical-reduction.x f-logical-reduction.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
