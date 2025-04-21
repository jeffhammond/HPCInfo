OPT	 = -O3 -fopenmp
FC	 = gfortran -ffree-form -fcheck=all -g3
CC	 = gcc -std=c11

CFLAGS   = $(OPT)
FFLAGS   = $(OPT)

LD       = $(FC)
LDFLAGS  = $(FFLAGS)
LIBS     =

all: before.x

%.x: parent.o child.o
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@

parent.o: parent.F90
	$(FC) $(FFLAGS) -c $< -o $@

child.o: child.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f a.out
	-rm -f *.o
	-rm -f *.mod
	-rm -f *.so
	-rm -f *.cod
	-rm -f $(TESTS)

