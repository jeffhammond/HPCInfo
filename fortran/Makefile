#OPT	 = -O3 -qopenmp
#FC	 = ifx -free #-std18
#CC	 = icx -std=c11

#OPT	 = -O3 # -fopenmp
#FC	 = gfortran -ffree-form -fcheck=all -g3 #-std=f2018 
#CC	 = gcc -std=c11

#OPT	 = -O3
#FC	 = ftn
#CC	 = cc

#OPT	 = -O3 -target=gpu
#FC	 = nvfortran
#CC	 = nvc

#CFLAGS   = $(OPT) -acc -mp
#FFLAGS   = $(OPT) -Mfree -cuda -acc -mp -stdpar -gpu=managed

LD       = $(FC)
LDFLAGS  = $(OPT)
LIBS     =

TESTS = bool ieee array reshape reshape3 reshape4 reshape4all reshape6 matmul atomic align # interop_f

all: $(TESTS)

interop_f: interop_f.o interop_c.o
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@

align: align.o my_allocate.o
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@

bool: bool.o print.o
	$(LD) $(LDFLAGS) $^ -o $@

extern: extern.o extern1.o extern2.o
	$(LD) $(LDFLAGS) $^ -o $@

%.o: %.F90
	$(FC) $(FFLAGS) -c $< -o $@

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.mod
	-rm -f *.so
	-rm -f *.cod
	-rm -f $(TESTS)

