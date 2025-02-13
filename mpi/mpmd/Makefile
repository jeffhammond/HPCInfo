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

%.x: %.F90 affinity.o
	$(FC) $(FFLAGS) $^ -o $@

affinity.o: cbin.c
	$(CC) $(CFLAGS) -DNO_MAIN -c $< -o $@

clean:
	-rm -f *.o *.x
