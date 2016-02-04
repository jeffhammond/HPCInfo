CC       = gcc-5
CFLAGS   = -O3 -Wall -std=c11
LIBS     =

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

TESTS = atomics ieee threads

all: $(TESTS)

%: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)


