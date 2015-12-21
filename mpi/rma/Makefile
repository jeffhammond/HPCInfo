OPT	 = -O3
CC       = mpicc
CFLAGS   = $(OPT) -std=c99 -Wall
LD       = $(CC)
LDFLAGS  = $(OPT)
LIBS     =

TESTS = gems

all: $(TESTS)

%: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

