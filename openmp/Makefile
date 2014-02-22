CC       = gcc
CFLAGS   = -g -O2 -fopenmp -std=gnu99
LD       = $(CC)
LDFLAGS  = $(CFLAGS)
LIBS     = 

all: matrix.x 

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x

distclean: realclean

