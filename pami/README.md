# Definition

PAMI stands for Parallel Active Messaging Interface and is produced by IBM.

# Documentation

## Papers

* [PAMI: A Parallel Active Message Interface for the Blue Gene/Q Supercomputer](http://www.computer.org/csdl/proceedings/ipdps/2012/4675/00/4675a763-abs.html)
* [Acceleration of an Asynchronous Message Driven Programming Paradigm on IBM Blue Gene/Q](http://charm.cs.illinois.edu/newPapers/12-50/paper.pdf) (about the Charm++ implementation on PAMI)

## User Community

Please sign up for the [PAMI discussion](http://lists.alcf.anl.gov/mailman/listinfo/pami-discuss) list to interact with the PAMI user community.

## API Documentation

See `pami.h`.

## Doxygen

EPFL hosts this at https://bgq1.epfl.ch/navigator/resources/doc/pami/index.html

## Programming Guide

[PAMI Programming Guide](http://publibfp.dhe.ibm.com/epubs/pdf/a2322733.pdf) was the first publicly available documentation of the PAMI API.  Please note that this document is about the AIX implementation of PAMI.  This document was originally full of errors and still may not be correct in all cases.  Please always verify syntax with `pami.h` on your system.

# Portability

PAMI is described on slides 13 and 14 of [Bob Wisniewski's Salishan 2011 talk](http://www.lanl.gov/conferences/salishan/salishan2011/1wisniewski.pdf).

# Source

PAMI is part of the driver that you can download by following the directions [here](https://wiki.alcf.anl.gov/parts/index.php/Blue_Gene/Q#Source_Code).

# Examples

I put all of my code online here: http://code.google.com/p/pami-examples/.  I do not intend to maintain the code listed below.

## Using multiple clients

PAMI supports multiple clients, e.g., MPI and ARMCI, MPI and UPC, MPI and CAF, etc.  Using a PAMI clients (e.g. MPI) with an SPI client (e.g. the lattice QCD codes) requires different resource control.

The client name 'MPI' is reserved by MPI and should not be used by any client that intends to inter-operate with the provided MPI library.

This test demonstrates how to use multiple clients:

```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <pami.h>

int main(int argc, char* argv[])
{
  pami_result_t result = PAMI_ERROR;

  /* initialize the client1 */
  char * client1name = "CLIENT1";
  pami_client_t client1;
  result = PAMI_Client_create(client1name, &client1, NULL, 0);
  assert(result == PAMI_SUCCESS);

  char * client2name = "CLIENT2";
  pami_client_t client2;
  result = PAMI_Client_create(client2name, &client2, NULL, 0);
  assert(result == PAMI_SUCCESS);

  /* finalize the client1 */
  result = PAMI_Client_destroy(&client2);
  assert(result == PAMI_SUCCESS);

  result = PAMI_Client_destroy(&client1);
  assert(result == PAMI_SUCCESS);

  printf("end of test \n");
  fflush(stdout);
  sleep(1);

  return 0;
}
```

You can submit with `qsub -t 5 -n 1 --mode c1 --env PAMI_CLIENTS=CLIENT1,CLIENT2 ./clients.x`.
