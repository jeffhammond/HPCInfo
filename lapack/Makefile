FC      := gfortran
FFLAGS   = -g -O2 
LIBS     = -framework Accelerate

LD       = $(FC)
LDFLAGS  = $(FFLAGS)

TESTS = test_eigen

all: $(TESTS)

%: %.f
	$(FC) $(FFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -rf *.dSYM
	-rm -f $(TESTS)

