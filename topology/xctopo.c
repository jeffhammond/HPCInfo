#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>

#include "xctopo.h"

int xctopo_get_mycoords(xctopo_t * topo)
{
  FILE * procfile = fopen("/proc/cray_xt/cname","r");
  if (procfile!=NULL) {

    char a, b, c, d;
    int col, row, cage, slot, anode;

    /* format example: c1-0c1s2n1 c3-0c2s15n3 */
    fscanf(procfile, 
           "%c%d-%d%c%d%c%d%c%d", 
           &a, &col, &row, &b, &cage, &c, &slot, &d, &anode);

#ifdef DEBUG
    fprintf(stderr, "coords = (%d,%d,%d,%d,%d) \n", rank, col, row, cage slot, anode);
#endif
    
    topo->col   = col;
    topo->row   = row;
    topo->cage  = cage;
    topo->slot  = slot;
    topo->anode = anode;

    fclose(procfile);

  } else {

    fprintf(stderr, "xctopo_get_mycoords: fopen has failed! \n");
    return 1;

  }

  return 0;
}
