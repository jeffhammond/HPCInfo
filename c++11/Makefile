# You might need to read http://libcxx.llvm.org/ and/or
# related Q&A on StackOverflow to get C++11 working.

CXX      = clang++ #-v
CXXFLAGS = -g -O2 -std=c++11 -stdlib=libc++
LIBS     = -nodefaultlibs -lc++ -lc++abi  -lm -lc

LD       = $(CXX)
LDFLAGS  = $(CXXFLAGS)

TESTS = future packaged_task threads lambda

all: $(TESTS)

%: %.cxx
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)
	-rm -rf *.dSYM

