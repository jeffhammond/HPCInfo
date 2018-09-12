#CXX		 = g++-8 -std=c++2a
CXX		 = clang++ -std=c++14

all: lock-free.x

%.x: %.cc
	$(CXX) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -f *.x
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt


