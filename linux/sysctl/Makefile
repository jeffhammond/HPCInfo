CC	= icc
CFLAGS	= -g -Wall -O2 -std=gnu99 -fopenmp

.PHONY: all clean

all: cpu_model

%.x: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f  *.o
	-rm -f  *.x
	-rm -fr *.dSYM
