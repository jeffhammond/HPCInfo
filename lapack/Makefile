FC       = gfortran
FFLAGS   = -O2 -ffast-math
LIBS     = -framework Accelerate

FC       = ifort
FFLAGS   = -Ofast
LIBS     = -mkl

LD       = $(FC)
LDFLAGS  = $(FFLAGS)

TESTS = test_dsygv test_dsyev

all: $(TESTS)

%: %.f90
	$(FC) $(FFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -rf *.dSYM
	-rm -f $(TESTS)

