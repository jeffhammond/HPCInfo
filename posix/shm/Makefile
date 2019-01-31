CC       = mpicc
CFLAGS   = -g -O2
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

TESTS = shm

all: $(TESTS)

%: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM


