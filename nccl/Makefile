CXX       = nvcc
CPPFLAGS  = 
CPPFLAGS += -I/usr/include/x86_64-linux-gnu/mpich
CXXFLAGS  = -g -O2 -std=c++20
CXXFLAGS += -arch=native
CXXFLAGS += $(CPPFLAGS)
LDFLAGS   = -lnccl -lcurand -lcublas
#LDFLAGS  += --forward-unknown-to-host-compiler -Wl,-Bsymbolic-functions -flto=auto -ffat-lto-objects -flto=auto -Wl,-z,relro -L/usr/lib/x86_64-linux-gnu -lmpich
LDFLAGS  += -L/usr/lib/x86_64-linux-gnu -lmpich

main: reduce.x

%.x: %.cu
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	-rm -f *.o *.x
