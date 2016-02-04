OPT	 := -O3 -qopenmp
FC	 := ifort -std08
FFLAGS   := $(OPT)
LD       := $(FC)
LDFLAGS  := $(OPT)
LIBS     :=

TESTS = ieee

all: $(TESTS)

%: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%: %.c
	$(CC) $(CFLAGS) $< -o $@

%: %.f90
	$(FC) $(FFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)

