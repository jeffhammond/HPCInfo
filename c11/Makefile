CC       = gcc-5
CFLAGS   = -g -O2 -std=c11 #-fopenmp
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

TESTS = atomics tls #threads

all: $(TESTS)

%: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)


