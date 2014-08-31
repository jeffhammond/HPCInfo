DC       = ldc2
DFLAGS   = -g -O2

all: hello

%: %.d
	$(DC) $(DFLAGS) $< 

clean:
	-rm -f *.o

realclean: clean
	-rm -f hello

distclean: realclean

