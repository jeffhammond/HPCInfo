#CC	 = icc
#CFLAGS	 = -O3
#CFLAGS	+= -qopenmp
#CFLAGS += -xHOST
#CFLAGS += -xCOMMON-AVX512
#CFLAGS += -xMIC-AVX512

#CC	 = gcc-11
CC	 = nvcc
#CC	 = clang#++
CFLAGS	 = -O3
#CFLAGS	+= -fopenmp
#CFLAGS	+= -D__AVX2__ -D__AVX__ -D__SSE4_1__ -D__SSE2__
CFLAGS	+= -D__SSE2__ -D__SSE4_1__
CFLAGS  += -I$(HOME)/simde/simde/x86 -DSIMDE
CFLAGS  += -DSIMDE_ENABLE_NATIVE_ALIASES

# Mac laptop
#LIBS += -Wl,-L/usr/local/Cellar/llvm/4.0.0/lib -Wl,-rpath -Wl,/usr/local/Cellar/llvm/4.0.0/lib

#CFLAGS	+= -std=gnu++11
#CFLAGS	+= -std=gnu11
#CFLAGS	+= -Wall
#CFLAGS	+= -g3

#CFLAGS	+= -DSTATIC_ALLOCATION

AS 	 = $(CC)
ASFLAGS	 = $(CFLAGS)
ASFLAGS += -S -fcode-asm -fverbose-asm -fasm-blocks -fsource-asm

BIN  = driver
OBJ  = copy.o stride.o util.o
INC  = copy.h stride.h util.h
INC += setup.h compiler.h

all: $(BIN)

driver: driver.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) $< $(OBJ) $(LIBS) -o $@

%.o: %.c $(INC)
	$(CC) $(CFLAGS) -c $< -o $@

%.s: %.c $(INC)
	$(AS) $(ASFLAGS) $< -o $@

archive: driver.c copy.c copy.h stride.c stride.h util.c util.h
	git archive --prefix=simd-memtest/ -o ../simd-memtest.tar.gz HEAD
clean:
	-rm -f *.s
	-rm -f $(OBJ)
	-rm -f $(BIN)
	-rm -fr *.dSYM

