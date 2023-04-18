CC      = mpicc
CFLAGS  = -g -O2 -Wall
#CFLAGS += -fopenmp

FC	= mpifort
FFLAGS  = -g -O2 -Wall

all: cbin.x fbin.x

%.x: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.x: %.F90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	-rm -f *.o *.x
