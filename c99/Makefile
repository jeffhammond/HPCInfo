CC      := icc
CFLAGS   = -g -O3 -Wall -std=c99
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

TESTS = array2d array3d restrict warn-stack array-restrict alias2d.o overflow

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt

