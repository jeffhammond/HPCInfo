CC	= icc
CFLAGS	= -g -Wall -O2

.PHONY: all clean

all: asmtest.x

asmtest.x: asmtest.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f  *.o
	-rm -f  *.x
	-rm -fr *.dSYM
