#CXX		=	g++-8
#CXXFLAGS	=	-g -Wall -O3 -fopenmp -mavx -mfma -march=haswell
#ASMFLAGS	=	-fverbose-asm
#CXX		=	/usr/local/Cellar/llvm/6.0.1/bin/clang++
#CXXFLAGS	=	-g -Wall -O3 -fopenmp -mavx -mfma -march=haswell
#ASMFLAGS	=	-fverbose-asm
CXX		=	icpc
CXXFLAGS	=	-g -Wall -O3 -qopenmp -xCORE-AVX2
#CXXFLAGS	=	-g -Wall -O3 -qopenmp -xCORE-AVX512 -qopt-zmm-usage=high
ASMFLAGS	=	-fsource-asm -fverbose-asm -fasm-blocks -fcode-asm

all: bfloat16 bfloat16.s

bfloat16: bfloat16.cc
	${CXX} ${CXXFLAGS} $< -o $@

bfloat16.s: bfloat16.cc
	${CXX} ${CXXFLAGS} -S ${ASMFLAGS} $< -o $@

clean:
	rm -f bfloat16
	rm -f *.o
	rm -f *.s
	rm -rf *.dSYM

