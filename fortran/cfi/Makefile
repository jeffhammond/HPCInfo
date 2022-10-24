CC = gcc-12
FC = gfortran-12

CFLAGS = -std=c11
FCFLAGS = -std=f2018

all: class.x

class.x: class.F90 foo.o
	$(FC) $(FCFLAGS) $^ -o $@

foo.o: foo.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -fr *.dSYM
	-rm -f *.x
	-rm -f *.mod
	-rm -f *.o
	-rm -f a.out

