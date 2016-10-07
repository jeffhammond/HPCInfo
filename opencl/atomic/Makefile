#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#     make clean    ... remove object and executable files.
#

# verify that you are including the right make.def file for the platform
include ../common/make.def

COMMON      = ../common

EXES     = vadd$(EXE) add$(EXE)


all: $(EXES)

vadd$(EXE): vadd.$(OBJ)  vadd.cl
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o vadd$(EXE) vadd.$(OBJ) $(LIBS)

add$(EXE): add.$(OBJ)  add.cl
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o add$(EXE) add.$(OBJ) $(LIBS)

test: $(EXES)
	$(PRE)vadd$(EXE);
	$(PRE)add$(EXE);

clean:
	$(RM) $(EXES) *.$(OBJ)

veryclean:
	$(RM) $(EXES)  *.$(OBJ)

.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<

