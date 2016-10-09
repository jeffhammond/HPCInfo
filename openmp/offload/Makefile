CC	= icc
CFLAGS	= -O3 -std=gnu99 -qopenmp
CFLAGS += -qopenmp-offload=gfx
CFLAGS += -g3 -Wall

LD	= $(CC)
LDFLAGS = $(CFLAGS)

LIBS	= #-static-intel

all: test_vadd.x

%.x: %.o
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

.PHONY: clean

clean:
	-rm -f *.o *.x

