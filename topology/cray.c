/* I think this is sufficient to address both intranode and internode topology, 
 * at least on XE systems.
 *
 * If you want detailed intranode topology information, 
 * use the <tt>xthi.c</tt> program in Section 9.3 of 
 * http://docs.cray.com/cgi-bin/craydoc.cgi?mode=View;id=S-2496-4002 
 *
 * I compiled this program with "cc -std=c99 crayxe_pmi.c -o crayxe_pmi.x" after doing
 * 'module load xtpe-network-gemini pmi rca'
 * If this doesn't work for you, try adding more system modules.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pmi.h>

int main(void)
{
  int rc;
  int rank, size;

  PMI_BOOL initialized;
  rc = PMI_Initialized(&initialized);
  if (rc!=PMI_SUCCESS)
    PMI_Abort(rc,"PMI_Initialized failed");

  if (initialized!=PMI_TRUE)
  {
    int spawned;
    rc = PMI_Init(&spawned);
    if (rc!=PMI_SUCCESS)
      PMI_Abort(rc,"PMI_Init failed");
  }

  rc = PMI_Get_rank(&rank);
  if (rc!=PMI_SUCCESS)
    PMI_Abort(rc,"PMI_Get_rank failed");

  rc = PMI_Get_size(&size);
  if (rc!=PMI_SUCCESS)
    PMI_Abort(rc,"PMI_Get_size failed");

  printf("rank %d of %d \n", rank, size);

  int rpn; /* rpn = ranks per node */
  rc = PMI_Get_clique_size(&rpn);
  if (rc!=PMI_SUCCESS)
    PMI_Abort(rc,"PMI_Get_clique_size failed");
  printf("rank %d clique size %d \n", rank, rpn);

  int * clique_ranks = malloc( rpn * sizeof(int) );
  if (clique_ranks==NULL)
    PMI_Abort(rpn,"malloc failed");

   rc = PMI_Get_clique_ranks(clique_ranks, rpn);
  if (rc!=PMI_SUCCESS)
    PMI_Abort(rc,"PMI_Get_clique_ranks failed");
  for(int i = 0; i<rpn; i++)
    printf("rank %d clique[%d] = %d \n", rank, i, clique_ranks[i]);

  int nid;
  rc = PMI_Get_nid(rank, &nid);
  if (rc!=PMI_SUCCESS)
    PMI_Abort(rc,"PMI_Get_nid failed");
  printf("rank %d PMI_Get_nid gives nid %d \n", rank, nid);

#if OLD
  rca_mesh_coord_t xyz;
  rca_get_meshcoord( (uint16_t) nid, &xyz);
  printf("rank %d rca_get_meshcoord returns (%2u,%2u,%2u)\n", 
         rank, xyz.mesh_x, xyz.mesh_y, xyz.mesh_z);
#else // UNTESTED
  pmi_mesh_coord_t xyz;
  PMI_Get_meshcoord((uint16_t) nid, &xyz);
  printf("rank %d PMI_Get_meshcoord returns (%2u,%2u,%2u)\n", 
         rank, xyz.mesh_x, xyz.mesh_y, xyz.mesh_z);
#endif

  fflush(stdout);
  return 0;
}
