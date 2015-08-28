OPT	 = -g -O2 -fopenmp
CC       = gcc-5
CFLAGS   = $(OPT) -std=gnu11
FC	 = gfortran-4.9
FFLAGS   = $(OPT)
LD       = $(CC)
LDFLAGS  = $(OPT)
LIBS     =

TESTS = matrix fusion nesting tls

all: $(TESTS)

%: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%: %.c
	$(CC) $(CFLAGS) $< -o $@

%: %.f90
	$(FC) $(FFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)

