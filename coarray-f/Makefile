# Cray XE6
FTN      = ftn -h caf
LIBS     = -lhugetlbfs

FTNFLAGS = -O2
LD       = $(FTN)
LDFLAGS  = $(FTNFLAGS)

all: hello.x atomic.x swap.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.f
	$(FTN) $(FTNFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x

distclean: realclean

