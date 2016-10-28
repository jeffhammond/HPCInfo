OFLAGS   := -O3 -fopenmp -Wall -fPIC

# Apple debug
# OFLAGS += -g3 -Wl,-pie

#OFLAGS   += -DSEQUENTIAL_CONSISTENCY

ifndef COMPILER
    COMPILER=intel
endif

ifeq ($(COMPILER),gcc)
    CC  = gcc
    CXX = g++
    # GCC on Mac is ancient and does not support OpenMP
    # This macro is from the NWChem build system
    IS_GCC4= $(shell $(CC) -dM -E - < /dev/null | egrep __VERS | cut -c22 | awk ' /6/  {print "N";exit}; /5/  {print "N";exit}; /3/  {print "N";exit}; /2/ {print "N";exit};{print "Y"}')
    ifeq ($(IS_GCC4),Y)
        # Homebrew on Mac
        CC  = gcc-6
        CXX = g++-6
    endif
    ASMFLAGS = -S -fverbose-asm
else ifeq ($(COMPILER),clang)
    # Linux
    CC  = clang
    CXX = clang++
    # Default Clang on Mac does not support OpenMP
    # This macro does not work as desired on Linux
    # This macro is from the NWChem build system
    IS_GCC4= $(shell $(CC) -dM -E - < /dev/null | egrep __VERS | cut -c22 | awk ' /6/  {print "N";exit}; /5/  {print "N";exit}; /3/  {print "N";exit}; /2/ {print "N";exit};{print "Y"}')
    ifeq ($(IS_GCC4),Y)
        # Homebrew on Mac
        CC  = clang-3.8
        CXX = clang++-3.8
        ifeq ($(CXX),clang++-3.8)
            OMPFLAGS  = -I/usr/local/Cellar/libiomp/20150701/include/libiomp
            # Clang may try to link libomp, so we symlink this to libiomp5
            OMPLIBS   = -L/usr/local/Cellar/libiomp/20150701/lib -liomp5
        endif
    endif
    ASMFLAGS  = -S -fverbose-asm
else ifeq ($(COMPILER),intel)
    CC       = icc
    CXX      = icpc
    ASMFLAGS = -S -fverbose-asm -fcode-asm -fasm-blocks
else
    CC       = false
    CXX      = false
endif

CFLAGS   := $(OFLAGS) $(OMPFLAGS) -std=gnu11
CXXFLAGS := $(OFLAGS) $(OMPFLAGS) -std=gnu++11

LIBS     = $(OMPLIBS)

LD       = $(CXX)
LDFLAGS  = $(OFLAGS)

TESTS = cxx11-counter.$(COMPILER) \
	cxx11-summation.$(COMPILER) \
	cxx11-nxtval.$(COMPILER) \
	cxx11-nxtval-vs-dynamic.$(COMPILER)

ASM =   cxx11-counter.$(COMPILER).s \
	cxx11-summation.$(COMPILER).s \
	cxx11-nxtval.$(COMPILER).s \
	cxx11-nxtval-vs-dynamic.$(COMPILER).s

all: $(TESTS)

asm: $(ASM)

# binaries

%.$(COMPILER): %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%.$(COMPILER): %.cc myatomics.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

# assembly

%.$(COMPILER).s: %.c
	$(CC) $(CFLAGS) $(ASMFLAGS) $< -o $@

%.$(COMPILER).s: %.cc myatomics.h
	$(CXX) $(CXXFLAGS) $(ASMFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt
	-rm -f a.out
	-rm -f *.gcc
	-rm -f *.clang
	-rm -f *.icc
	-rm -f *.intel


