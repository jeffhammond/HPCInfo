# Cray XE6
#FTN     := ftn -h caf
#LIBS    := -lhugetlbfs
# Intel
#FTN     := ifort -free -std08
#LIBS    :=
# GCC
FTN     := gfortran -ffree-form -fcoarray=single
#FTN     := gfortran-6 -ffree-form -fcoarray=single
LIBS    := -lcaf_single

FTNFLAGS = -O2
LD       = $(FTN)
LDFLAGS  = $(FTNFLAGS)

all: hello.x atomic.x swap.x gems.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.f
	$(FTN) $(FTNFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x


