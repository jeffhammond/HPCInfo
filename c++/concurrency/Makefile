# You might need to read http://libcxx.llvm.org/ and/or
# related Q&A on StackOverflow to get C++11 working.

CXX      = /usr/local/Cellar/llvm/7.0.1/bin/clang++
CXXFLAGS = -g -O2 -std=c++11 -stdlib=libc++
LIBS     = -nodefaultlibs -lc++ -lc++abi  -lm -lc
#CXX      = icpc
#CXXFLAGS = -g -O2 -std=c++11
#LIBS     = 

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

