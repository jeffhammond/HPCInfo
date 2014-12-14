## Language Standard 

See the [UPC specification repo on Google Code](http://code.google.com/p/upc-specification/).

## Implementations 

* Cray has a high-quality UPC implementation in their compiler toolchain.  Obviously, you need to own a Cray supercomputer to have the Cray UPC compiler.  The Cray UPC compiler targets [[DMAPP]] as the low-level runtime system.

* [Berkeley UPC](http://upc.lbl.gov/) is based upon source-to-source translation of UPC to C with UPCR (UPC Runtime) calls.  Berkeley UPCR uses [GASNet](http://gasnet.cs.berkeley.edu/), which is a high-quality PGAS and active-message runtime library that runs on almost every machine known to me.

* [GCC UPC](http://www.gccupc.org/), or GUPC, is a high-quality UPC compiler that uses the Berkeley UPCR or their own shared-memory runtime.

* Other vendors support UPC, including IBM and HP, but I have no experience with those.

## Documentation

[Berkeley UPC documentation](http://upc.lbl.gov/docs/)

[GWU UPC page](http://upc.gwu.edu/)

## Sample Programs 

If you have a [Cray](https://github.com/jeffhammond/HPCInfo/wiki/Cray) machine and do ```module load PrgEnv-cray```, you can use the Makefile.

See [NERSC's PGAS page](http://www.nersc.gov/users/computational-systems/hopper/programming/PGAS/) for details about running on Hopper.
