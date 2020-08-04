FC		=	ifort
FCFLAGS		=	-g -O3 -qopenmp -xCORE-AVX2 -mkl

all: test.x

test.x: test.F
	$(FC) $(FCFLAGS) test.F -o test.x

clean:
	rm -f test.x
	rm -f *.o
	rm -f *.s
	rm -rf *.dSYM

