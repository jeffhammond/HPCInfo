CC := nvc
CFLAGS := -std=c11

FC := nvfortran
FCFLAGS := -O2
FCFLAGS += -gpu=managed
FCFLAGS += -cudalib=nvblas
FCFLAGS += -cudalib=nvlamath

# required on nuclear (x86_64)
#CFLAGS  += -fPIE
#FCFLAGS += -fPIE

all: test.x

test.x: test.F90
	$(FC) $(FCFLAGS) $^ -o $@

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

clean:
	-rm -f *.o *.mod

make realclean: clean
	-rm -f *.x
