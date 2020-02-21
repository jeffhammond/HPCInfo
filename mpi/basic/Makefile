CC      = mpicc
COPT    = -g -O2 -Wall #-std=gnu99

# OpenMP
# modify this as appropriate if you're compiler is not GCC
#COPT += -fopenmp

LD      = $(CC)
CFLAGS  = $(COPT) -DCHECK_ERRORS

LDFLAGS = $(COPT)
LIBS    = -lm

all: hello.x reduce.x bcast.x montecarlo.x rsend.x

extra: mpi-omp.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.x
