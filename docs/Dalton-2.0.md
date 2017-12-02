This pertains to an older version of Dalton.  Please upgrade to the latest Dalton release.

## Overview

Dalton is a quantum chemistry program developed primarily in Europe which emphasizes molecular properties.  It was not designed to run on supercomputers but achieves reasonably scalability for the subset of features which are parallelized perform fairly well.  Only the DFT code runs in parallel (in direct mode) so you should not attempt to run the MCSCF or CC methods on BGP.  Parallel CC properties are available in NWChem.

You can find instructions on how to license Dalton from the [Dalton home page](http://daltonprogram.org/).  Community support is often all that is available but the quality of the code and extensive documentation make usage rather straightforward.  For BGP-specific problems with Dalton, you may contact me after you have determined that you are using Dalton correctly (your exact input runs correctly on another platform).

## Building Dalton

### Blue Gene/P

To configure Dalton on BGP, you should start use the configure script for an AIX system: `./configure -aix` in the top Dalton source directory (i.e. `/home/$USER/dalton-2.0`).  Accept all the default options even though they are wrong, then manually change them in Makefile.config.  These are the settings in Makefile.config which can be used to successfully build Dalton on Intrepid:

```
ARCH        = rs6000
#
CPPFLAGS      = -WF,-DSYS_AIX,-DVAR_MFDS,-DVAR_SPLITFILES,-D'INSTALL_WRKMEM=600000000',-D'INSTALL_BASDIR="/home/$USER/dalton-2.0/basis/"',-DVAR_MPI,-DIMPLICIT_NONE
F77           = mpixlf77_r
CC            = mpixlc_r
RM            = rm -f
FFLAGS        = -O5 -qarch=450d -qtune=450 -qextname -qessl
SAFEFFLAGS    = -O3 -qstrict -qarch=450d -qtune=450 -qextname -qessl
CFLAGS        = -DVAR_MPI -I../include -DSYS_AIX -D_LARGE_FILES -qlanglvl=stdc99 -DRESTRICT=restrict -O3 -qstrict -qarch=450 -qtune=450
INCLUDES      = -I../include
LIBS          = -L/soft/apps/ESSL-4.4.1-0/lib -lesslbg
INSTALLDIR    = /home/$USER/dalton-2.0/bin
PDPACK_EXTRAS = linpack.o eispack.o gp_dlapack.o gp_zlapack.o gp_dblas3.o gp_dblas2.o gp_dblas1.o gp_zblas.o
AR            = ar
ARFLAGS       = rvs
#
MPI_INCLUDE_DIR = /bgsys/drivers/ppcfloor/comm/include
MPI_LIB_PATH    = -L/bgsys/drivers/ppcfloor/comm/lib
MPI_LIB         = -lfmpich_.cnk -lmpich.cnk -ldcmf.cnk -ldcmfcoll.cnk -lpthread -lrt -L/bgsys/drivers/ppcfloor/runtime/SPI -lSPI.cna
```

You need to modify `cc/crayio.c` by adding the following three lines at the top of the file:

```c
/* BlueGene/P https://wiki.alcf.anl.gov/index.php/Compiling_and_linking */
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
```

You also need to modify `gp/gphjj.F` by commenting out the following lines with `cc` and adding the second line (technically optional):

```c
#if defined (SYS_AIX)
      WRITE(LUPRI,*) 'BGP has no obvious system traceback facility.'
C  920522-hjaaj -- ad hoc routine for creating traceback on IBM-AIX
C  Note: integer divide by zero is the only error which
C        always will cause an exception
C
cc      include 'fexcp.h'
cc      save i,j
cc      data i,j /1,0/
C
cc      call SIGNAL(SIGTRAP,xl__trce)
cc      i = i/j
#endif
```

Manual compilation of `amfi/amfi.F` is necessary because lower optimization is necessary.  After issuing the following command in `dalton-2.0/amfi`, go back to the Dalton top directory and complete the build.

```
mpixlf77_r -I../include -WF,-DSYS_AIX,-DVAR_MFDS,-D'INSTALL_WRKMEM=60000000',-D'INSTALL_BASDIR="/home/$USER/dalton-2.0/basis/"',-DVAR_MPI,-DIMPLICIT_NONE \
-O3 -qarch=450d -qtune=450 -qextname -c amfi.F
```

These instructions are correct as of June 2, 2009.

### Cray XE6

* `crayftn` does not include cwd by default so one must add -I. to INCLUDES in `Makefile.config`.
* `sirius/koopro4.F` cannot be compiled with `-O2` using `crayftn` but `-O1` works.
* `crayftn` leads to link errors from MPICH code to GNI and XPMEM so I will try PGI instead.

This is the `Makefile.config` leads to a successful build with PGI compilers.  Correct execution not yet verified.

```
ARCH        = linux
#
#
CPPFLAGS      = -DSYS_LINUX -DVAR_PGF77 -DVAR_MFDS -D'INSTALL_WRKMEM=100000000' -D'INSTALL_BASDIR="/users/jhammond/DALTON/dalton-2.0-cam/basis/"' -DVAR_MPI -DIMPLICIT_NONE
F77           = ftn
CC            = cc
RM            = rm -f
FFLAGS        = -O3
SAFEFFLAGS    = -O2
CFLAGS        = -O2 -DRESTRICT=restrict
INCLUDES      = -I../include  -I.
LIBS          = #-L/opt/xt-libsci/10.4.9/cray/lib -lsci_mc12
INSTALLDIR    = /users/jhammond/DALTON/dalton-2.0-cam/bin
PDPACK_EXTRAS = linpack.o eispack.o
GP_EXTRAS     =
AR            = ar
ARFLAGS       = rvs
#
default : linuxparallel.x
#
# Parallel initialization
#
MPI_INCLUDE_DIR =
MPI_LIB_PATH    =
MPI_LIB         =
#
#
# Suffix rules
# hjaaj Oct 04: .g is a "cheat" suffix, for debugging.
#               'make x.g' will create x.o from x.F or x.c with -g debug flag set.
#
.SUFFIXES : .F .o .c .i .g

.F.o:
        $(F77) $(INCLUDES) $(CPPFLAGS) $(FFLAGS) -c $*.F

.F.g:
        $(F77) $(INCLUDES) $(CPPFLAGS) $(FFLAGS) -g -c $*.F

.c.o:
        $(CC) $(INCLUDES) $(CPPFLAGS) $(CFLAGS) -c $*.c

.c.g:
        $(CC) $(INCLUDES) $(CPPFLAGS) $(CFLAGS) -g -c $*.c

.F.i:
        $(F77) $(INCLUDES) $(CPPFLAGS) -E $*.F > $*.i
```

I am using these modules:

```
  1) modules                                                                   12) Base-opts/1.0.2-1.0301.21771.3.3.gem
  2) nodestat/2.2-1.0301.22648.3.3.gem                                         13) xtpe-network-gemini
  3) sdb/1.0-1.0301.22744.3.24.gem                                             14) pgi/10.9.0
  4) MySQL/5.0.64-1.0301.2899.20.4.gem                                         15) totalview-support/1.1.1
  5) lustre-cray_gem_s/1.8.2_2.6.27.48_0.1.1_1.0301.5475.7.1-1.0301.23312.0.0  16) xt-totalview/8.8.0a
  6) udreg/1.3-1.0301.2236.3.6.gem                                             17) xt-libsci/10.4.9
  7) ugni/2.0-1.0301.2365.3.6.gem                                              18) pmi/1.0-1.0000.8160.39.2.gem
  8) gni-headers/2.0-1.0301.2497.4.1.gem                                       19) xt-mpt/5.1.2
  9) dmapp/2.2-1.0301.2427.3.8.gem                                             20) /opt/cray/xt-asyncpe/4.5/modulefiles/xtpe-mc12
 10) xpmem/0.1-2.0301.22550.3.6.gem                                            21) xt-asyncpe/4.5
 11) slurm                                                                     22) PrgEnv-pgi/3.1.37AA

```

### POWER7

* `-O3 -qtune/qarch=auto` leads to XLF segfault for `pdpack/scatter-io.c` while `-O3 -qtune/qarch=pwr7` is fine, which has to be a bug.  ''Update: this has been fixed.''
* `cc/cc_hyppol.F` compilation fails with -O4 or -O5 but -O3 works.
* Linking with -O5 takes an extremely long time.

This `Makefile.config` works for serial execution.  Using the Linux target did not work.

```
ARCH        = rs6000
#
#
CPPFLAGS      = -WF,-DSYS_AIX,-DVAR_MFDS,-D_FILE_OFFSET_BITS=64,-D'INSTALL_WRKMEM=100000000',-D'INSTALL_BASDIR="/home/jhammond/DALTON/dalton-2.0-cam/basis/"',-DIMPLICIT_NONE
F77           = xlf_r
CC            = xlc_r
RM            = rm -f
FFLAGS        = -O3 -qstrict -qarch=pwr7 -qtune=pwr7 -qmaxmem=-1 -qextname -q64
SAFEFFLAGS    = -O3 -qstrict -qarch=pwr7 -qtune=pwr7 -qmaxmem=-1 -qextname -q64
CFLAGS        = -I../include -DSYS_AIX -D_LARGE_FILES -qlanglvl=stdc99 -DRESTRICT=restrict -O3 -qarch=pwr7 -qtune=pwr7 -q64
INCLUDES      = -I../include
LIBS          =
INSTALLDIR    = /home/jhammond/DALTON/dalton-2.0-cam/bin
PDPACK_EXTRAS = linpack.o eispack.o gp_dlapack.o gp_zlapack.o gp_dblas3.o gp_dblas2.o gp_dblas1.o gp_zblas.o
GP_EXTRAS     =
AR            = ar
ARFLAGS       = rvs
# flags for ftnchek on Dalton /hjaaj
CHEKFLAGS  = -nopure -nopretty -nocommon -nousage -noarray -notruncation -quiet  -noargumants -arguments=number  -usage=var-unitialized
# -usage=var-unitialized:arg-const-modified:arg-alias
# -usage=var-unitialized:var-set-unused:arg-unused:arg-const-modified:arg-alias
#
default : dalton.x
#
# Suffix rules
#
.SUFFIXES : .F .o .c .i

.F.o:
        $(F77) $(INCLUDES) $(CPPFLAGS) $(FFLAGS) -c $*.F

.c.o:
        $(CC) $(CFLAGS) -c $*.c

.F.i:
        $(F77) $(INCLUDES) $(CPPFLAGS) -E $*.F > $*.i
```

## Running Jobs

The Dalton run script does not work with the BGP queue system.  You must either modify the script to work with the queue system (create two scripts - one for running and one for clean-up) or use the binary directly.

Actually, one should be able to run the Dalton script as a script job within Cobalt (or other scheduler).  I just haven't verified that it works.  On ALCF systems, one needs to use <tt>cobalt-mpirun</tt>.

## Known Bugs

* Some of the Dalton test jobs fail on many platforms, particularly those related to geometry derivatives.  If most of the test jobs pass, your binary is probably good.
* Due to file naming conventions, one cannot run Dalton on 1000+ nodes.  Hence, the biggest job you should run on BGP is 999 nodes on a 1024-node partition.  Running on 512 nodes is fine, of course.  '''THIS BUG HAS BEEN FIXED'''.  If necessary, please ask Jeff for the patch.
* The Dalton 2.0 patched release which includes CAM-B3LYP contains a bug in the quadratic response function only for parallel execution.  Polarizabilities will be correct but hyperpolarizabilities are not.  TDDFT excited-states (poles of the linear response function) should be correct but two-photon properties are likely wrong but neither has been tested.  The developers are aware of this problem but have not provided a solution.
* PBE and PBE0 are '''wrong''' for properties.  See the user list for details.

## Patches

http://www.scalalife.eu/content/dalton-downloads