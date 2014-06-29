FC      = ftn 
CC      = cc 
COPT    = -g -O2
FOPT    = -g -O2

CFLAGS  = $(COPT) $(INCLUDE)

LD      = $(CC)
LDFLAGS = $(COPT)

all: test.x

%.x: %.o xctopo.o
	$(LD) $(LDFLAGS) $< xctopo.o -o $@

%.o: %.c xctopo.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(RMFLAGS) *.o

realclean: clean
	$(RM) $(RMFLAGS) *.a *.x

