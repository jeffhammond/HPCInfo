CC	= icc
CFLAGS	= -O3 -std=gnu99 -qopenmp
CFLAGS += -qopenmp-offload=gfx
CFLAGS += -g3 -Wall
LIBS	= #-static-intel

#CC	= gcc-7
#CFLAGS	= -O3 -std=gnu99 -fopenmp -fcilkplus
#CFLAGS += -g3 -Wall

LD	= $(CC)
LDFLAGS = $(CFLAGS)

all: test_vadd.x

%.x: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

.PHONY: clean

clean:
	-rm -f *.o *.x

