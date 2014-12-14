CC       = gcc-4.9
CFLAGS   = -g -O2 -std=c11
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

TESTS = atomics #threads

all: $(TESTS)

%: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)


