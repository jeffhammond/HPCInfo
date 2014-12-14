CC      = mpicc
CFLAGS  = -g -O2 -std=c99 -Wall -Werror

LD      = $(CC)
LDFLAGS = $(CFLAGS)

TESTS = win_lock

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -rf *.dSYM
	-rm -f $(TESTS)

