# Cray XE6
UPC      = cc -h upc
LIBS     = -lhugetlbfs
# Jeff's Mac
#UPC		 = upc 

UPCFLAGS = -O2 -DRELAXED
LD       = $(UPC)
LDFLAGS  = $(UPCFLAGS)

all: hello.x barrier.x shared.x example.x gems.x # matvec.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.upc
	$(UPC) $(UPCFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -rf external/

