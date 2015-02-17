CC       = cc
CFLAGS   = -O2

LD       = $(CC)
LDFLAGS  = $(CFLAGS)

TESTS = put get amo

all: $(TESTS)

%: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)


