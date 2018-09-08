CC	 = gcc
FC       = gfortran
OFLAGS    = -O2 -ffast-math
LIBS     = -framework Accelerate

CC	 = icc
FC       = ifort
OFLAGS   = -Ofast
LIBS     = -mkl -DMKL

CFLAGS   = $(OFLAGS) -std=c99 -DDEBUG
FFLAGS   = $(OFLAGS)

TESTS = test_dsygv test_dsyev test_xgemm

all: $(TESTS)

%: %.f90
	$(FC) $(FFLAGS) $< $(LIBS) -o $@

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -rf *.dSYM
	-rm -f $(TESTS)

