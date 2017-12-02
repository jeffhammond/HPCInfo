## Proxies

This documentation assumes you know what a proxy is and why you need it.

### Proxifier

[Proxifier](http://www.proxifier.com/mac/) is a very nice GUI for proxies that supports rules to selectively apply different proxies for different applications, servers and/or ports.  It is not free (as in beer).

I setup Proxifier for use with Intel's [Endeavor](http://www.top500.org/system/176908) in Proxifier as follows.

Proxy
* Address = proxy.jf.intel.com
* Protocol = SOCKS Version 5
* Do not set to default.

Rules
* Name = SSH
* Applications = ssh
* Target Hosts = ```<IP ADDRESS as X.Y.Z.*>```
* Target Ports = NULL (Any)
* Action = Proxy SOCKS5...

### Netcat

The following works.  Note that Netcat options may not be backwards-compatible, so this may not work forever.

```
  Host NAME
    Hostname <IP ADDRESS>
    ProxyCommand nc -x proxy-us.intel.com:1080 %h %p
```

### Connect

The following has not been tested.

```
   Host NAME
     Hostname <IP ADDRESS>
     ProxyCommand $SOME_PATH/connect -S proxy-socks.sc.intel.com %h %p
```

You need to download [connect.c](http://foo-projects.org/~sofar/connect.c) and compile it into ```$SOME_PATH```. *This documentation is not an endorsement of this code nor should it be interpreted as a statement that the code is free from security vulnerabilities.  You are absolutely on your own in this respect.*

## BLAS and LAPACK

```LIBS=-framework Accelerate```.  Read ```man Accelerate``` for more information.

## Homebrew

- http://brew.sh/
- http://blog.shvetsov.com/2014/11/homebrew-cheat-sheet-and-workflow.html
- http://www.commandlinefu.com/commands/view/4831/update-all-packages-installed-via-homebrew
```
brew update && brew install `brew outdated`
```

### GCC

```
brew install gcc49 --enable-fortran --without-multilib
```

### LLVM

```
brew install --HEAD llvm35 --with-clang --with-libcxx --rtti
# works around bug reported already
brew install --HEAD isl && brew install --HEAD cloog && brew install -v --HEAD llvm35 --with-clang --with-libcxx --with-asan --rtti
```

### Cleanup

```
brew cleanup -ns # dry run
brew cleanup -s # for real
```



## Macports

**I abandoned Macports completely after it switched my ABI to i386 without asking and broke absolutely everything horribly.  Now I use Homebrew instead.**

- http://superuser.com/questions/80168/how-to-uninstall-all-unused-versions-of-a-macports-package-at-once
- http://stackoverflow.com/questions/10319314/prevent-macports-from-installing-pre-built-package (in particular <tt>port -s -v install package +options</tt>)

### GCC

- http://stackoverflow.com/questions/8361002/how-to-use-the-gcc-installed-in-macports
- http://superuser.com/questions/423254/macports-gcc-select-error-trying-to-exec-i686-apple-darwin11-llvm-gcc-4-2

### MPICH

I forget stuff like this...

```
sudo port install mpich +gcc47
```

```
export VERSION=dev

# GCC

export GCC_VERSION=-4.8

../configure CC=gcc${GCC_VERSION} CXX=g++${GCC_VERSION} FC=gfortran${GCC_VERSION} F77=gfortran${GCC_VERSION} --enable-fortran --enable-threads=runtime --enable-g=dbg --with-pm=hydra --prefix=/opt/mpich/$VERSION/gcc/default --enable-wrapper-rpath --enable-static --enable-shared && make install

../configure CC=gcc${GCC_VERSION} CXX=g++${GCC_VERSION} FC=gfortran${GCC_VERSION} F77=gfortran${GCC_VERSION} --enable-fortran --enable-threads=runtime --enable-g=all --with-pm=hydra --prefix=/opt/mpich/$VERSION/gcc/debug --enable-wrapper-rpath --enable-static --enable-shared --enable-nemesis-dbg-localoddeven && make install

../configure CC=gcc${GCC_VERSION} CXX=g++${GCC_VERSION} FC=gfortran${GCC_VERSION} F77=gfortran${GCC_VERSION} --enable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/gcc/fast --enable-static --enable-fast=O3,nochkmsg,notiming,ndebug,nompit --disable-weak-symbols --enable-threads=funneled --enable-wrapper-rpath --enable-static --enable-shared && make install

# LLVM

../configure CC=clang CXX=clang++ FC=false F77=false --disable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/clang/debug --enable-threads=runtime --enable-g=all --enable-wrapper-rpath --enable-static --enable-shared --enable-nemesis-dbg-localoddeven && make install

../configure CC=clang CXX=clang++ FC=false F77=false --disable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/clang/fast --enable-fast=O3,nochkmsg,notiming,ndebug,nompit --disable-weak-symbols --enable-threads=funneled --enable-wrapper-rpath --enable-static --enable-shared && make install

../configure CC=clang CXX=clang++ FC=false F77=false --disable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/clang/default --enable-wrapper-rpath --enable-static --enable-shared && make install

# INTEL

../configure CC=icc CXX=icpc FC=ifort F77=ifort --enable-fortran --enable-threads=runtime --enable-g=dbg --with-pm=hydra --prefix=/opt/mpich/$VERSION/intel/default --enable-wrapper-rpath --enable-static --enable-shared && make install

../configure CC=icc CXX=icpc FC=ifort F77=ifort --enable-fortran --enable-threads=runtime --enable-g=all --with-pm=hydra --prefix=/opt/mpich/$VERSION/intel/debug --enable-wrapper-rpath --enable-static --enable-shared --enable-nemesis-dbg-localoddeven && make install

../configure CC=icc CXX=icpc FC=ifort F77=ifort --enable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/intel/fast --enable-static --enable-fast=O3,nochkmsg,notiming,ndebug,nompit --disable-weak-symbols --enable-threads=funneled --enable-wrapper-rpath --enable-static --enable-shared && make install
```

### Octave

Octave requires ARPACK, which defaults to using OpenMPI, but I want to use MPICH for everything.  I also prefer Accelerate to ATLAS, if for no other reason than ATLAS takes forever to build.

```
port install arpack +mpich -openmpi
port install octave +accelerate -atlas
```