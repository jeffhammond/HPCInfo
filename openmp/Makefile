OPT	     = -g -O2 -fopenmp
CC       = gcc-4.9
CFLAGS   = $(OPT) -std=gnu99
FC	 	 = gfortran-4.9
FFLAGS   = $(OPT)
LD       = $(CC)
LDFLAGS  = $(OPT)
LIBS     = 

TESTS = matrix fusion nesting

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

