CXX		 = clang++
CXXFLAGS	 = -std=c++11

all: lock-free.x

%.x: %.cc
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -f *.x
	-rm -rf *.dSYM
	-rm -f *.dwarf
	-rm -f *.optrpt


