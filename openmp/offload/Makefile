CC	= icc
CFLAGS	= -O3 -std=gnu99 -qopenmp
#CFLAGS += -qopenmp-offload=gfx
CFLAGS += -qopenmp-offload=host
CFLAGS += -g3 -Wall
#CFLAGS += -DUSE_GFX # not working
LIBS	= #-static-intel

#CC	= gcc-6
#CFLAGS	= -O3 -std=gnu99 -fopenmp -fcilkplus
#CFLAGS += -g3 -Wall

#CC	= clang-3.9
#CFLAGS	= -O3 -std=gnu99 -fopenmp=libiomp5
#CFLAGS += -g3 -Wall
#CFLAGS += -omptargets=x86_64-pc-linux-gnu

LD	= $(CC)
LDFLAGS = $(CFLAGS)

all: test_vadd.x test_syscall.x test_memory_model.x

%.x: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

.PHONY: clean

clean:
	-rm -f a.out
	-rm -f *.o
	-rm -f *.x
	-rm -rf *.dSYM

