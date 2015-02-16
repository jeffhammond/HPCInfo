## Background

Coarrays are part of the Fortran standard.  See http://www.co-array.org/ for details.

John Mellor-Crummey's group at Rice has an alternative view about Coarray Fortran that is summarized on [Wikipedia](http://en.wikipedia.org/wiki/Coarray_Fortran).

I found ftp://ftp.numerical.rl.ac.uk/pub/talks/jkr.reading.5XI08.pdf and http://www.co-array.org/caf_intro.htm helpful when writing examples.

## Implementations

Coarrays are part of the Fortran 2008 standard and must be supported by any F08-compliant compiler.

GCC 5.0 is supposed to have an implementation of Fortran 2008 coarrays based upon [OpenCoarrays](http://opencoarrays.org/), to which I have contributed some patches.

Cray's Fortran compiler supports CAF and the performance is known to be good.  The Intel Fortran compiler also supports coarrays but the performance quality has been questioned (see ["Evaluation of the Coarray Fortran Programming Model on the Example of a Lattice Boltzmann Code"](https://sites.google.com/a/lbl.gov/pgas12/home/contributed-papers)).

The [OpenUH](http://www2.cs.uh.edu/~openuh/) compiler is an open-source implementation of CAF.  Rice provides an open-source implementation of [CAF 2.0](http://caf.rice.edu/download.html).

## Example Code 

On Cray systems, do this first:

XE6:
```
module swap PrgEnv-pgi PrgEnv-cray
```
XC30:
```
module swap PrgEnv-intel PrgEnv-cray
```
