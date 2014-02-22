# Cray XE6
UPC      = cc -h upc
LIBS     = -lhugetlbfs
# Jeff's Mac
#UPC		 = upc 

UPCFLAGS = -g -O2 -DRELAXED
LD       = $(UPC)
LDFLAGS  = $(UPCFLAGS)

all: hello.x barrier.x shared.x example.x matvec.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.upc
	$(UPC) $(UPCFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x
	-rm -rf external/

distclean: realclean

