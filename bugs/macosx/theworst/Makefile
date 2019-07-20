all: sandbox1 sandbox2

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f sandbox1 sandbox2
	-rm -f *.o


