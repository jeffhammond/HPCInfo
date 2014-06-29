CC      = mpicc
COPT    = -g -O2 -std=gnu99 -fopenmp

LD      = $(CC)
CFLAGS  = $(COPT)
LDFLAGS = $(COPT) bgq_threadid.o -lm -lpthread

all: mpi_omp_pthreads.x

%.x: %.o bgq_threadid.o
	$(LD) $(LDFLAGS) $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(RMFLAGS) *.o *.lst 

realclean: clean
	$(RM) $(RMFLAGS) *.x

