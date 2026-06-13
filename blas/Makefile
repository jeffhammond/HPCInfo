FC	 = gfortran
FFLAGS = -fdefault-integer-8

all: blas_wrapper.o gemm_wrapper.o

%.o: %.F90
	$(FC) $(FFLAGS) -c $^ -o $@

clean:
	-rm -f *.o
	-rm -f *.mod
	-rm -f *.x

