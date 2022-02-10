#CXX = nvc++ --c++20 -acc -I./libcudacxx/include
CXX = clang++-12 -std=c++20 -fopenmp

all: basic.x

%.x: %.cc
	$(CXX) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -f *.x
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt


