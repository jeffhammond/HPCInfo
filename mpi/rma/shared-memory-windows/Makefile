RMA_SYNC_MODE := -1

CC      = mpicc
CFLAGS  = -g -O2 -Wall -DRMA_SYNC_MODE=$(RMA_SYNC_MODE)

LD      = $(CC)
LDFLAGS = $(CFLAGS)

all: main.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x
