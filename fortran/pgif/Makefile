CC := nvc
CFLAGS := -std=c11
CFLAGS += -I/opt/nvidia/hpc_sdk/Linux_$$(uname -m)/22.7/compilers/include_acc

FC := nvfortran
#FCFLAGS := -std=f2018 $(WARNFLAGS)

# required on nuclear (x86_64)
CFLAGS  += -fPIE
FCFLAGS += -fPIE

all: test_cfi.x

test_cfi.x: test_cfi.F90 foo_cfi.c
	$(CC) $(CFLAGS) -c foo_cfi.c -o foo_cfi.o
	$(FC) $(FCFLAGS) test_cfi.F90 foo_cfi.o -o test_cfi.x

clean:
	-rm -f test_cfi.x test_cfi.o foo_cfi.o cfi.mod
