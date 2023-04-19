GCC	= gcc-12
CC      = OMPI_CC=${GCC} MPICH_CC=${GCC} mpicc
CFLAGS  = -g -O2 -Wall
CFLAGS += -fopenmp

FC	= mpifort
FFLAGS  = -g -O2 -Wall
FFLAGS += -fopenmp

all: cbin.x fbin.x

%.x: %.c
	$(CC) $(CFLAGS) $< -o $@

%.x: %.F90
	$(FC) $(FFLAGS) $< -o $@

clean:
	-rm -f *.o *.x
