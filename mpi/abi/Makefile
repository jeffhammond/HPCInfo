CC      = gcc-13
CFLAGS  = -g -O3 -Wall -std=c11

FC	= gfortran-13
FFLAGS  = -g -O3 -Wall -std=f2018

LD      = $(CC)
LDFLAGS = -O3

# both of the following need VAPAA
FFLAGS += -I/Users/jhammond/Work/MPI/vapaa/source
LIBS    = -L/Users/jhammond/Work/MPI/vapaa/source -lmpi_f08
# MUK
#CFLAGS += -I/Users/jhammond/Work/MPI/mukautuva
#LIBS   += -L/Users/jhammond/Work/MPI/mukautuva -lmuk
# MPICH no fortran
CFLAGS += -I/opt/mpich/no-fortran/include
LIBS   += -L/opt/mpich/no-fortran/lib -lmpi -lpmpi

# MPICH fortran
#CFLAGS += -I/opt/mpich/fortran/include
#FFLAGS += -I/opt/mpich/fortran/include
#LIBS   += -L/opt/mpich/fortran/lib -lmpifort -lmpi -lpmpi

LIBS   += -L/opt/homebrew/Cellar/gcc/13.2.0/lib/gcc/13 -lgfortran
LIBS   += -lstdc++

all: c-logical-reduction.x f-logical-reduction.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
