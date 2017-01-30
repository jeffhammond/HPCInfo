#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#     make clean    ... remove object and executable files.
#

# verify that you are including the right make.def file for the platform
include ../common/make.def

COMMON      = ../common

EXES     = vadd$(EXE)


all: $(EXES)

vadd$(EXE): vadd.cpp.$(OBJ) vadd.cl
	$(CLINKER) $(CFLAGS) vadd.$(OBJ) $(LIBS) $(OPENCLFLAGS) -o vadd$(EXE)

test: $(EXES)
	$(PRE)vadd$(EXE);

clean:
	$(RM) $(EXES) *.$(OBJ)

veryclean:
	$(RM) $(EXES)  *.$(OBJ)

.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

%.c.$(OBJ): %.c ../common/cl.hpp
	$(CC) $(CFLAGS) -c $<

%.cpp.$(OBJ): %.cpp ../common/cl.hpp
	$(CC) $(CFLAGS) -c $<

