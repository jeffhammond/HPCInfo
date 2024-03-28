CC      = clang
CFLAGS  = -g -O2 -Wall -std=c11
CFLAGS += -I/Users/jhammond/Work/MPI/mukautuva

FC	= gfortran
FFLAGS  = -g -O2 -Wall -std=f2018
FFLAGS += -I/Users/jhammond/Work/MPI/vapaa/source

LD      = $(CC)
LDFLAGS =
LIBS    = -L/Users/jhammond/Work/MPI/vapaa/source -lmpi_f08
LIBS   += -L/opt/homebrew/Cellar/gcc/13.2.0/lib/gcc/13 -lgfortran
LIBS   += -L/Users/jhammond/Work/MPI/mukautuva -lmuk
LIBS   += -lc++

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
