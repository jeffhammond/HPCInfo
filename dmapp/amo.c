/*
 *  Copyright (C) 2010 by Argonne National Laboratory.
 *
 * dmapp_amo.c
 *
 *  Created on: Sep 18, 2010
 *      Author: jeff
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#ifdef __CRAYXE
#include "pmi.h"
#include "dmapp.h"
#endif

int main(int argc, char **argv)
{
#ifdef __CRAYXE
    int i,j;
    int me = -1;
    int size = -1;
    //int fail_count = 0;

    dmapp_return_t status;
    dmapp_rma_attrs_t actual_args = { 0 }, rma_args = { 0 };
    dmapp_jobinfo_t job;
    dmapp_seg_desc_t *seg = NULL;

    /* Set the RMA parameters. */
    rma_args.put_relaxed_ordering = DMAPP_ROUTING_ADAPTIVE;
    rma_args.max_outstanding_nb = DMAPP_DEF_OUTSTANDING_NB;
    rma_args.offload_threshold = DMAPP_OFFLOAD_THRESHOLD;
    rma_args.max_concurrency = 1;

    /* Initialize DMAPP. */
    status = dmapp_init(&rma_args, &actual_args);
    assert(status==DMAPP_RC_SUCCESS);

    /* Get job related information. */
    status = dmapp_get_jobinfo(&job);
    assert(status==DMAPP_RC_SUCCESS);

    me = job.pe;
    size = job.npes;
    seg = &(job.sheap_seg);

    /* Allocate and initialize the source and target arrays. */
    long * source = (long *) dmapp_sheap_malloc( size * sizeof(long) );
    assert(source!=NULL);
    long * target = (long *) dmapp_sheap_malloc( size * sizeof(long) );
    assert(target!=NULL);

    for (i = 0; i < size; i++) source[i] = 0;
    for (i = 0; i < size; i++) target[i] = 0;

    /* Wait for all PEs to complete array initialization. */
    PMI_Barrier();

    /* compare-and-swap */
    //
    // dmapp_return_t dmapp_acswap_qw(
    //   IN void             *target_addr /* local memory */,
    //   IN void             *source_addr /* remote memory */,
    //   IN dmapp_seg_desc_t *source_seg  /* remote segment */,
    //   IN dmapp_pe_t        source_pe   /* remote rank */,
    //   IN int64_t           comperand,
    //   IN int64_t           swaperand);
    //
    for (i = 0; i < size; i++)
        if (i != me)
        {
            status = dmapp_acswap_qw(&source[i], &target[i], seg, (dmapp_pe_t)i, (int64_t)0, (int64_t)me);
            if (status==DMAPP_RC_SUCCESS)                printf("%d: DMAPP_RC_SUCCESS\n",me);
            else if (status==DMAPP_RC_INVALID_PARAM)     printf("%d: DMAPP_RC_INVALID_PARAM\n",me);
            else if (status==DMAPP_RC_ALIGNMENT_ERROR)   printf("%d: DMAPP_RC_ALIGNMENT_ERROR\n",me);
            else if (status==DMAPP_RC_NO_SPACE)          printf("%d: DMAPP_RC_NO_SPACE\n",me);
            else if (status==DMAPP_RC_TRANSACTION_ERROR) printf("%d: DMAPP_RC_TRANSACTION_ERROR\n",me);
            fflush(stdout);
            assert(status==DMAPP_RC_SUCCESS);
        }

    /* Wait for all PEs. */
    PMI_Barrier();

    /* see who won */
    for (i = 0; i < size; i++)
    {
        if (i==me)
        {
            for (j = 0; j < size; j++) printf("me = %d target[%d] = %ld\n", me, i, target[i] );
            printf("==========================================\n");
            fflush(stdout);
        }
        PMI_Barrier();
    }

    /* Finalize. */
    status = dmapp_finalize();
    assert(status==DMAPP_RC_SUCCESS);

#endif
    return(0);
}
