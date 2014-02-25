# You might need to read http://libcxx.llvm.org/ and/or
# related Q&A on StackOverflow to get C++11 working.

CXX      = clang++-3.4 #-v
CXXFLAGS = -g -O2 -std=c++11 -stdlib=libc++
LIBS     = -nodefaultlibs -lc++ -lc++abi  -lm -lc

LD       = $(CXX)
LDFLAGS  = $(CXXFLAGS)

all: future.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x

distclean: realclean

