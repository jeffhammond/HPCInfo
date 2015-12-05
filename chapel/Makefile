CHPL      = chpl
CHPLFLAGS =

all: gems.x

%.x: %.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

clean:
	-rm -f *.x

