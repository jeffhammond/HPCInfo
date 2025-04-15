#OPT	 = -O3 -qopenmp
#FC	 = ifx -free #-std18
#CC	 = icx -std=c11

#OPT	 = -O3 # -fopenmp
#FC	 = gfortran -ffree-form -fcheck=all -g3 #-std=f2018 
#CC	 = gcc -std=c11 -DGFORTRAN

#OPT	 = -O3
#FC	 = ftn
#CC	 = cc

OPT	 = -O3
FC	 = nvfortran
CC	 = nvc

CFLAGS   = $(OPT) -acc -mp -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/include/
FFLAGS   = $(OPT) -Mfree #-cuda -acc -mp -stdpar -gpu=managed -target=gpu

LD       = $(FC)
LDFLAGS  = $(FFLAGS)
LIBS     =

TESTS = bool logical logical_value ieee \
	array reshape reshape3 reshape4 reshape4all reshape6 \
	matmul atomic align extern # interop_f

all: $(TESTS)

interop_f: interop_f.o interop_c.o
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@

align: align.o my_allocate.o
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@

bool: bool.o print.o
	$(LD) $(LDFLAGS) $^ -o $@

logical_value: logical_value.o set_logical.o
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
	-rm -f a.out
	-rm -f *.o
	-rm -f *.mod
	-rm -f *.so
	-rm -f *.cod
	-rm -f $(TESTS)

