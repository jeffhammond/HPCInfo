CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -Werror

all: to_chars.x

%.x: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f *.o *.x *.s
	-rm rf *.dSYM

