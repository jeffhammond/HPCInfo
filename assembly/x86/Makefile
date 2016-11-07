CC	= icc
CFLAGS	= -g -Wall -O2 -std=gnu99 -fopenmp

.PHONY: all clean

all: mov.x pause.x

%.x: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f  *.o
	-rm -f  *.x
	-rm -fr *.dSYM
