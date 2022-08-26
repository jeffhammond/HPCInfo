CC := nvc
CFLAGS := -std=c11
#CFLAGS += -I/opt/nvidia/hpc_sdk/Linux_$$(uname -m)/22.7/compilers/include_acc

FC := nvfortran
FCFLAGS := -O3
#FCFLAGS += -Mstack_arrays
#FCFLAGS += -acc=gpu

# required on nuclear (x86_64)
#CFLAGS  += -fPIE
#FCFLAGS += -fPIE

all: test_cfi.x test_pgif.x

test_cfi.x: test_cfi.F90 foo_cfi.o
	$(FC) $(FCFLAGS) $^ -o $@

%.o: %.c pgif90.h
	$(CC) $(CFLAGS) -c $< -o $@

test_pgif.x: test_pgif.c foo_pgif.o
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

clean:
	-rm -f *.x *.o *.mod
