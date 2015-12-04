OPT	 = -O3 -qopenmp
CC       = icc
CFLAGS   = $(OPT) -std=c99
FC	 = ifort
FFLAGS   = $(OPT)
LD       = $(CC)
LDFLAGS  = $(OPT)
LIBS     =

TESTS = matrix fusion nesting tls gems # distribute

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

