CC = gcc-14
FC = gfortran-14

CFLAGS = -g -std=c11
FCFLAGS = -g -std=f2018

all: class.x array.x

%.x: %.F90 foo.o
	$(FC) $(FCFLAGS) $^ -o $@

foo.o: foo.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -fr *.dSYM
	-rm -f *.x
	-rm -f *.mod
	-rm -f *.o
	-rm -f a.out

