OPT	 = -g -O2 -fopenmp
CC       = gcc-4.9
CFLAGS   = $(OPT) -std=gnu99
FC	 = gfortran-4.9
FFLAGS   = $(OPT)
LD       = $(CC)
LDFLAGS  = $(OPT)
LIBS     = 

all: matrix.x fusion.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.x: %.c 
	$(CC) $(CFLAGS) $< -o $@

%.x: %.f90
	$(FC) $(FFLAGS) $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x

distclean: realclean

