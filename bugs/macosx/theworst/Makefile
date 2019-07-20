CC = gcc-9

all: sandbox1 sandbox2

%: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f sandbox1 sandbox2
	-rm -f *.o


