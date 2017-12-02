# Overview

* http://en.wikipedia.org/wiki/Malloc#Implementations
* [The Hole That dlmalloc Can’t Fill](http://gameangst.com/?p=496)

# Implementations

## Doug Lea's Malloc (dlmalloc)

* http://gee.cs.oswego.edu/dl/html/malloc.html

## Wolfram Gloger's Malloc (ptmalloc)

* http://www.malloc.de/en/

## Jason Evan's Malloc (jemalloc)

* http://www.canonware.com/jemalloc/
* https://github.com/jemalloc/jemalloc

### Building jemalloc on Blue Gene/Q

I think this works now.
```
cd $HOME/MALLOC
git clone git://canonware.com/jemalloc.git
cd jemalloc
./autogen.sh
./configure CC=powerpc64-bgq-linux-gcc \
            --host=powerpc64-bgq-linux \
            --prefix=$HOME/MALLOC/jemalloc-install \
            --disable-valgrind
make CC="powerpc64-bgq-linux-gcc -Wl,-shared -shared" -j16
make CC="powerpc64-bgq-linux-gcc -Wl,-shared -shared" check
make install
```

If you do not want to override `CC`, the edit you need to make is in line 11 of `Makefile`:
```
CC := powerpc64-bgq-linux-gcc -Wl,-shared -shared
```

## Google's Malloc (tcmalloc)

* Docs: http://goog-perftools.sourceforge.net/doc/tcmalloc.html
* Source: https://code.google.com/p/gperftools/

## Hoard Malloc

* Home: http://www.hoard.org/
* Source: https://github.com/emeryberger/Hoard

## Concur Malloc

* http://sourceforge.net/projects/concur/

## Niall Douglas' Malloc (nedmalloc)

* http://www.nedprod.com/programs/portable/nedmalloc/
* http://sourceforge.net/projects/nedmalloc/

## SFMalloc

* Home: http://aces.snu.ac.kr/Center_for_Manycore_Programming/SFMalloc.html
* Source (unofficial): https://github.com/jeffhammond/sfmalloc is my personal fork

## DMalloc

* Home: http://dmalloc.com/