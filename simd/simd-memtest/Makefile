CC	 = icc
CFLAGS	 = -O3
CFLAGS	+= -qopenmp
#CFLAGS += -xHOST
#CFLAGS += -xCOMMON-AVX512
CFLAGS += -xMIC-AVX512
#CFLAGS += -xCORE-AVX512

#CC	 = g++-6
#CC	 = clang++-4.0
#CFLAGS	 = -O3
#CFLAGS	+= -fopenmp
#CFLAGS	+= -march=core-avx2

# Mac laptop
#LIBS += -Wl,-L/usr/local/Cellar/llvm/4.0.0/lib -Wl,-rpath -Wl,/usr/local/Cellar/llvm/4.0.0/lib

#CFLAGS	+= -std=gnu++11
CFLAGS	+= -std=gnu11
CFLAGS	+= -Wall
CFLAGS	+= -g3

CFLAGS	+= -DSTATIC_ALLOCATION -mcmodel=medium

AS 	 = $(CC)
ASFLAGS	 = $(CFLAGS)
ASFLAGS += -S -fcode-asm -fverbose-asm -fasm-blocks -fsource-asm

BIN  = driver
OBJ  = copy.o triad.o stride.o util.o
INC  = copy.h triad.h stride.h util.h
INC += setup.h compiler.h

all: $(BIN)

driver: driver.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) $< $(OBJ) $(LIBS) -o $@

%.o: %.c $(INC)
	$(CC) $(CFLAGS) -c $< -o $@

%.s: %.c $(INC)
	$(AS) $(ASFLAGS) $< -o $@

archive: driver.c copy.c copy.h triad.c triad.h stride.c stride.h util.c util.h
	git archive --prefix=simd-memtest/ -o ../simd-memtest.tar.gz HEAD
clean:
	-rm -f *.s
	-rm -f $(OBJ)
	-rm -f $(BIN)
	-rm -fr *.dSYM

