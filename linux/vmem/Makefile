OFLAGS ?= -Os -Wall -fPIC -std=gnu99

# Apple debug
# OFLAGS += -g3 -Wl,-pie
OFLAGS += -g3

ifndef COMPILER
    COMPILER = gcc
endif

ifeq ($(COMPILER),gcc)
CC        = gcc
ASMFLAGS  = -S -fverbose-asm
else ifeq ($(COMPILER),clang)
CC        = clang-3.8
ASMFLAGS  = -S -fverbose-asm
# Clang tries to link libomp, so we symlink this to libiomp5
else ifeq ($(COMPILER),intel)
CC        = icc
ASMFLAGS  = -S -fverbose-asm -fcode-asm -fasm-blocks
else
CC        = false
endif

CFLAGS   := $(OFLAGS) $(OMPFLAGS)

LIBS     := $(OMPLIBS)

LD       := $(CC)
LDFLAGS  := $(OFLAGS)

TESTS = test-mprotect.x test-canonical-addressing.x

all: $(TESTS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.x: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt
	-rm -f a.out
	-rm -f *.x

