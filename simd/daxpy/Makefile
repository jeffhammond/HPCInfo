#FC		=	ifort
#FCFLAGS		=	-g -O3 -qopenmp -xCORE-AVX2 -mkl -free -heap-arrays
FC		=	gfortran
FCFLAGS		=	-g -O3 -fopenmp -ffree-form  -ffree-line-length-none -Wall -std=f2008 -framework Accelerate -mavx

all: test.x

test.x: test.F
	$(FC) $(FCFLAGS) test.F -o test.x

clean:
	rm -f test.x
	rm -f *.o
	rm -f *.s
	rm -rf *.dSYM

