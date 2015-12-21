CHPL      = chpl
CHPLFLAGS =

all: gems.x exit.x

%.x: %.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

clean:
	-rm -f *.x

