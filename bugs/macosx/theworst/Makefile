CC = gcc-9

all: good bad

%: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f good bad
	-rm -f *.o


