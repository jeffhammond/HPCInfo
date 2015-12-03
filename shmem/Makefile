CC       = cc
CFLAGS   = -g -O2
LD       = $(CC)
LDFLAGS  = $(CFLAGS)
LIBS     = -lhugetlbfs

all: hello.x sheap.x get.x put.x gems.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c myshmem.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x

distclean: realclean

