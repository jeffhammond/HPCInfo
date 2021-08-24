FC = gfortran
FFLAGS = -O3 -std=f2008
LDFLAGS = -framework Accelerate

all: ccsd_aosig.x

ccsd_aosig.x: ccsd_aosig.F
	$(FC) $(FFLAGS) $< $(LDFLAGS) -o $@

clean:
	-rm -f *.o
	-rm -f *.x

