DMAPP is a one-sided communication API designed by Cray that looks a lot like [[SHMEM]].  It is the basis for implementing SHMEM as well as the Cray [[UPC]] and [[CAF]] runtimes.

# API Documentation

DMAPP documentation: 
* Version 3103 - [PDF](http://docs.cray.com/books/S-2446-3103/S-2446-3103.pdf) [HTML](http://docs.cray.com/cgi-bin/craydoc.cgi?mode=View;id=S-2446-3103)
* Version 4101 - [PDF](http://docs.cray.com/books/S-2446-4101/S-2446-4101.pdf) [HTML](http://docs.cray.com/cgi-bin/craydoc.cgi?mode=View;id=S-2446-4101)
* Version 5002 - [PDF](http://docs.cray.com/books/S-2446-5002/S-2446-5002.pdf) [HTML](http://docs.cray.com/cgi-bin/craydoc.cgi?mode=View;id=S-2446-5002)
* Version 51 - [PDF](http://docs.cray.com/books/S-2446-51/S-2446-51.pdf) [HTML](http://docs.cray.com/cgi-bin/craydoc.cgi?mode=View;id=S-2446-51)
* Version 52 - [PDF](http://docs.cray.com/books/S-2446-51/S-2446-52.pdf) [HTML](http://docs.cray.com/cgi-bin/craydoc.cgi?mode=View;id=S-2446-52)

# Example Code

You need to increase the symmetric heap size for these tests to run, e.g. using: ```export XT_SYMMETRIC_HEAP_SIZE=400M```
