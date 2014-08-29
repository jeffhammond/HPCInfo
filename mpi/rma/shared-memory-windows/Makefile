CC      = mpicc
CFLAGS  = -g -O2 -Wall -Werror

LD      = $(CC)
LDFLAGS = $(CFLAGS)

all: win_sync.x win_lock_exclusive.x win_lock_shared.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x
	-rm -rf *.dSYM
