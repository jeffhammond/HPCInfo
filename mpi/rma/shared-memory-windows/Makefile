CC      = mpicc
CFLAGS  = -g -O2 -std=c99 -Wall -Werror

LD      = $(CC)
LDFLAGS = $(CFLAGS)

all: win_fence.x win_pscw.x win_sync.x win_lock_exclusive.x win_lock_shared.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
	-rm -rf *.dSYM
