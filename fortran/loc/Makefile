FC	 = gfortran
CC	 = gcc
CFLAGS   = -std=c11
CFLAGS  += -I/opt/homebrew/Cellar/gcc/12.2.0/lib/gcc/current/gcc/aarch64-apple-darwin21/12/include

all: loc_f.x

loc_f.x: loc_f.F90 loc_c.o
	$(FC) $(FFLAGS) $^ -o $@

loc_c.o: loc_c.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f *.mod
	-rm -f *.x

