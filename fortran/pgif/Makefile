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

all: test_cfi.x

test_cfi.x: test_cfi.F90 foo_cfi.o
	$(FC) $(FCFLAGS) $^ -o $@

%.o: %.c pgif90.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f test_cfi.x test_cfi.o foo_cfi.o cfi.mod
