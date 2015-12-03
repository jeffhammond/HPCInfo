# Cray XE6
#FTN      = ftn -h caf
#LIBS     = -lhugetlbfs
# Intel
FTN      = ifort -free -std08
LIBS     =

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


