OPT	 = -O3 -qopenmp
FC	 = ifort -std18 -free
CC	 = icc -std=c11

OPT	 = -O3 # -fopenmp
FC	 = gfortran -std=f2018 -ffree-form
CC	 = gcc -std=c11

OPT	 = -O3
FC	 = nvfortran
CC	 = nvc

CFLAGS   = $(OPT) -acc -mp
FFLAGS   = $(OPT) -Mfree -cuda -acc -mp -stdpar

#LD       = $(FC)
#LDFLAGS  = $(OPT)
#LIBS     =

TESTS = ieee array reshape reshape3 reshape4 reshape6 matmul atomic # interop_f

all: $(TESTS)

interop_f: interop_f.o interop_c.o
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@

%: %.c
	$(CC) $(CFLAGS) $< -o $@

%: %.f90
	$(FC) $(FFLAGS) $< -o $@

%: %.F90
	$(FC) $(FFLAGS) $< -o $@

#%: %.o
#	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.mod
	-rm -f $(TESTS)

