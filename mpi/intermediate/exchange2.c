#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#include <mpi.h>
#if defined() || defined()
#  include <mpix.h>
#else
#  warning This test should be run on a Blue Gene.
#endif

#include "safemalloc.h"

int main(int argc, char *argv[])
{
    int provided;
    int world_size, world_rank;
    int node_size;

    MPI_Init_thread( &argc, &argv, MPI_THREAD_SINGLE, &provided );
    assert( provided>=MPI_THREAD_SINGLE );

    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    assert(world_size>1);

    int max_links = ( argc > 1 ? atoi(argv[1]) : 10 );
    int max_count = ( argc > 2 ? atoi(argv[2]) : 11*1024*1024 );
    //int nbrecv    = ( argc > 3 ? atoi(argv[3]) : 0 );

    int rank_ap = -1;
    int rank_am = -1;
    int rank_bp = -1;
    int rank_bm = -1;
    int rank_cp = -1;
    int rank_cm = -1;
    int rank_dp = -1;
    int rank_dm = -1;
    int rank_ep = -1;
    int rank_em = -1;
    //int rank_tp = -1;
    //int rank_tm = -1;

#if defined ()
    uint32_t xSize, ySize, zSize, tSize;
    uint32_t xRank, yRank, zRank, tRank;

    MPIX_rank2torus( world_size-1, &xSize, &ySize, &zSize, &tSize );
    MPIX_rank2torus( world_rank, &xRank, &yRank, &zRank, &tRank );

    node_size = tRank+1;
    MPI_Allreduce( MPI_IN_PLACE, &node_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    rank_ap = MPIX_torus2rank(xRank+1, yRank  , zRank  , tRank);
    rank_am = MPIX_torus2rank(xRank-1, yRank  , zRank  , tRank);
    rank_bp = MPIX_torus2rank(xRank  , yRank+1, zRank  , tRank);
    rank_bm = MPIX_torus2rank(xRank  , yRank-1, zRank  , tRank);
    rank_cp = MPIX_torus2rank(xRank  , yRank  , zRank+1, tRank);
    rank_cm = MPIX_torus2rank(xRank  , yRank  , zRank-1, tRank);

    if (world_rank==0) printf("# torus size %d = (%d,%d,%d) \n", world_size, xSize+1, ySize+1, zSize+1 );
    printf("# from %d (%d,%d,%d,%d) to A+/- (%d, %d), B+/- (%d, %d), C+/- (%d, %d) \n",
           world_rank, xRank, yRank, zRank, tRank,
           rank_ap, rank_am, rank_bp, rank_bm, rank_cp, rank_cm);
#elif defined()
    MPIX_Hardware_t hw;
    MPIX_Hardware(&hw);

    node_size = hw.ppn;

    {
        int hopCoord = (hw.Coords[0]+1) % (hw.Size[0]);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hopCoord, hw.Coords[1], hw.Coords[2], hw.Coords[3], hw.Coords[4], hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_ap);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_ap = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_ap);
    }
    {
        int hopCoord = (hw.Coords[0]==0) ? (hw.Size[0]-1) : (hw.Coords[0]-1);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hopCoord, hw.Coords[1], hw.Coords[2], hw.Coords[3], hw.Coords[4], hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_am);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_am = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_am);
    }
    {
        int hopCoord = (hw.Coords[1]+1) % (hw.Size[1]);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hw.Coords[0], hopCoord, hw.Coords[2], hw.Coords[3], hw.Coords[4], hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_bp);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_bp = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_bp);
    }
    {
        int hopCoord = (hw.Coords[1]==0) ? (hw.Size[1]-1) : (hw.Coords[1]-1);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hw.Coords[0], hopCoord, hw.Coords[2], hw.Coords[3], hw.Coords[4], hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_bm);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_bm = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_bm);
    }
    {
        int hopCoord = (hw.Coords[2]+1) % (hw.Size[2]);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hw.Coords[0], hw.Coords[1], hopCoord, hw.Coords[3], hw.Coords[4], hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_cp);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_cp = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_cp);
    }
    {
        int hopCoord = (hw.Coords[2]==0) ? (hw.Size[2]-1) : (hw.Coords[2]-1);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hw.Coords[0], hw.Coords[1], hopCoord, hw.Coords[3], hw.Coords[4], hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_cm);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_cm = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_cm);
    }
    {
        int hopCoord = (hw.Coords[3]+1) % (hw.Size[3]);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hw.Coords[0], hw.Coords[1], hw.Coords[2], hopCoord, hw.Coords[4], hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_dp);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_dp = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_dp);
    }
    {
        int hopCoord = (hw.Coords[3]==0) ? (hw.Size[3]-1) : (hw.Coords[3]-1);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hw.Coords[0], hw.Coords[1], hw.Coords[2], hopCoord, hw.Coords[4], hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_dm);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_dm = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_dm);
    }
    {
        int hopCoord = (hw.Coords[4]+1) % (hw.Size[4]);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hw.Coords[0], hw.Coords[1], hw.Coords[2], hw.Coords[3], hopCoord, hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_ep);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_ep = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_ep);
    }
    {
        int hopCoord = (hw.Coords[4]==0) ? (hw.Size[4]-1) : (hw.Coords[4]-1);
        int tempCoords[MPIX_TORUS_MAX_DIMS+1] = { hw.Coords[0], hw.Coords[1], hw.Coords[2], hw.Coords[3], hopCoord, hw.coreID};
        MPIX_Torus2rank(tempCoords, &rank_em);
        //printf("%d: tempCoords = (%d,%d,%d,%d,%d) rank_em = %d \n", world_rank, tempCoords[0], tempCoords[1], tempCoords[2], tempCoords[3], tempCoords[4], rank_em);
    }

    if (world_rank==0) printf("# torus size = %d (%d,%d,%d,%d,%d,%d) \n", world_size, hw.Size[0], hw.Size[1], hw.Size[2], hw.Size[3], hw.Size[4], hw.ppn);
    printf("# from %d (%d,%d,%d,%d,%d,%d) to A+/- (%d, %d), B+/- (%d, %d), C+/- (%d, %d), D+/- (%d, %d), E+/- (%d, %d) \n",
           world_rank, hw.Coords[0], hw.Coords[1], hw.Coords[2], hw.Coords[3], hw.Coords[4], hw.coreID,
           rank_ap, rank_am, rank_bp, rank_bm, rank_cp, rank_cm, rank_dp, rank_dm, rank_ep, rank_em);
#else
    rank_ap = (world_rank+1)%world_size;
    printf("#send from %d to %d \n", world_rank, rank_ap );
#endif
    fflush( stdout );
    MPI_Barrier( MPI_COMM_WORLD );
    //MPI_Abort( MPI_COMM_WORLD , 37);

    int * rbuf_ap = safemalloc(max_count * sizeof(int));
    int * rbuf_am = safemalloc(max_count * sizeof(int));
    int * rbuf_bp = safemalloc(max_count * sizeof(int));
    int * rbuf_bm = safemalloc(max_count * sizeof(int));
    int * rbuf_cp = safemalloc(max_count * sizeof(int));
    int * rbuf_cm = safemalloc(max_count * sizeof(int));
    int * rbuf_dp = safemalloc(max_count * sizeof(int));
    int * rbuf_dm = safemalloc(max_count * sizeof(int));
    int * rbuf_ep = safemalloc(max_count * sizeof(int));
    int * rbuf_em = safemalloc(max_count * sizeof(int));

    int * sbuf_ap = safemalloc(max_count * sizeof(int));
    int * sbuf_am = safemalloc(max_count * sizeof(int));
    int * sbuf_bp = safemalloc(max_count * sizeof(int));
    int * sbuf_bm = safemalloc(max_count * sizeof(int));
    int * sbuf_cp = safemalloc(max_count * sizeof(int));
    int * sbuf_cm = safemalloc(max_count * sizeof(int));
    int * sbuf_dp = safemalloc(max_count * sizeof(int));
    int * sbuf_dm = safemalloc(max_count * sizeof(int));
    int * sbuf_ep = safemalloc(max_count * sizeof(int));
    int * sbuf_em = safemalloc(max_count * sizeof(int));

    int tag_ap = 0;
    int tag_am = 1;
    int tag_bp = 2;
    int tag_bm = 3;
    int tag_cp = 4;
    int tag_cm = 5;
    int tag_dp = 6;
    int tag_dm = 7;
    int tag_ep = 8;
    int tag_em = 9;

    for ( int i = 0 ; i < max_count ; i++) sbuf_ap[i] = tag_ap;
    for ( int i = 0 ; i < max_count ; i++) sbuf_am[i] = tag_am;
    for ( int i = 0 ; i < max_count ; i++) sbuf_bp[i] = tag_bp;
    for ( int i = 0 ; i < max_count ; i++) sbuf_bm[i] = tag_bm;
    for ( int i = 0 ; i < max_count ; i++) sbuf_cp[i] = tag_cp;
    for ( int i = 0 ; i < max_count ; i++) sbuf_cm[i] = tag_cm;
    for ( int i = 0 ; i < max_count ; i++) sbuf_dp[i] = tag_dp;
    for ( int i = 0 ; i < max_count ; i++) sbuf_dm[i] = tag_dm;
    for ( int i = 0 ; i < max_count ; i++) sbuf_ep[i] = tag_ep;
    for ( int i = 0 ; i < max_count ; i++) sbuf_em[i] = tag_em;

    if (world_rank==0) printf("#begin nonblocking send-recv 5d halo exchange test\n" );

    fflush( stdout );
    MPI_Barrier( MPI_COMM_WORLD );

    for ( int count = 1; count <= max_count ; count *= 2 )
    {
        double dt[11];
        //for (int j=0;j<11;j++) dt[j] = 1.0e9;

        MPI_Request req[20];

        /* links = 0 is warmup on all 10 links */
        for ( int links = 0; links <= max_links ; links++ )
        { 
            for ( int i = 0 ; i < count ; i++) rbuf_ap[i] = 0;
            for ( int i = 0 ; i < count ; i++) rbuf_am[i] = 0;
            for ( int i = 0 ; i < count ; i++) rbuf_bp[i] = 0;
            for ( int i = 0 ; i < count ; i++) rbuf_bm[i] = 0;
            for ( int i = 0 ; i < count ; i++) rbuf_cp[i] = 0;
            for ( int i = 0 ; i < count ; i++) rbuf_cm[i] = 0;
            for ( int i = 0 ; i < count ; i++) rbuf_dp[i] = 0;
            for ( int i = 0 ; i < count ; i++) rbuf_dm[i] = 0;
            for ( int i = 0 ; i < count ; i++) rbuf_ep[i] = 0;
            for ( int i = 0 ; i < count ; i++) rbuf_em[i] = 0;

            MPI_Barrier( MPI_COMM_WORLD );

            double t0 = MPI_Wtime();

            if (links==0 || links>0)
            {
                MPI_Irecv( rbuf_ap, count, MPI_INT, rank_am, tag_ap, MPI_COMM_WORLD, &req[0] );
                MPI_Isend( sbuf_ap, count, MPI_INT, rank_ap, tag_ap, MPI_COMM_WORLD, &req[1] );
            }
            if (links==0 || links>1)
            {
                MPI_Irecv( rbuf_ap, count, MPI_INT, rank_ap, tag_am, MPI_COMM_WORLD, &req[2] );
                MPI_Isend( sbuf_ap, count, MPI_INT, rank_am, tag_am, MPI_COMM_WORLD, &req[3] );
            }
            if (links==0 || links>2)
            {
                MPI_Irecv( rbuf_bp, count, MPI_INT, rank_bm, tag_bp, MPI_COMM_WORLD, &req[4] );
                MPI_Isend( sbuf_bp, count, MPI_INT, rank_bp, tag_bp, MPI_COMM_WORLD, &req[5] );
            }
            if (links==0 || links>3)
            {
                MPI_Irecv( rbuf_bp, count, MPI_INT, rank_bp, tag_bm, MPI_COMM_WORLD, &req[6] );
                MPI_Isend( sbuf_bp, count, MPI_INT, rank_bm, tag_bm, MPI_COMM_WORLD, &req[7] );
            }
            if (links==0 || links>4)
            {
                MPI_Irecv( rbuf_cp, count, MPI_INT, rank_cm, tag_cp, MPI_COMM_WORLD, &req[8] );
                MPI_Isend( sbuf_cp, count, MPI_INT, rank_cp, tag_cp, MPI_COMM_WORLD, &req[9] );
            }
            if (links==0 || links>5)
            {
                MPI_Irecv( rbuf_cp, count, MPI_INT, rank_cp, tag_cm, MPI_COMM_WORLD, &req[10] );
                MPI_Isend( sbuf_cp, count, MPI_INT, rank_cm, tag_cm, MPI_COMM_WORLD, &req[11] );
            }
            if (links==0 || links>6)
            {
                MPI_Irecv( rbuf_dp, count, MPI_INT, rank_dm, tag_dp, MPI_COMM_WORLD, &req[12] );
                MPI_Isend( sbuf_dp, count, MPI_INT, rank_dp, tag_dp, MPI_COMM_WORLD, &req[13] );
            }
            if (links==0 || links>7)
            {
                MPI_Irecv( rbuf_dp, count, MPI_INT, rank_dp, tag_dm, MPI_COMM_WORLD, &req[14] );
                MPI_Isend( sbuf_dp, count, MPI_INT, rank_dm, tag_dm, MPI_COMM_WORLD, &req[15] );
            }
            if (links==0 || links>8)
            {
                MPI_Irecv( rbuf_ep, count, MPI_INT, rank_em, tag_ep, MPI_COMM_WORLD, &req[16] );
                MPI_Isend( sbuf_ep, count, MPI_INT, rank_ep, tag_ep, MPI_COMM_WORLD, &req[17] );
            }
            if (links==0 || links>9)
            {
                MPI_Irecv( rbuf_ep, count, MPI_INT, rank_ep, tag_em, MPI_COMM_WORLD, &req[18] );
                MPI_Isend( sbuf_ep, count, MPI_INT, rank_em, tag_em, MPI_COMM_WORLD, &req[19] );
            }

            if (links>0) MPI_Waitall( 2*links, req, MPI_STATUSES_IGNORE );
            else         MPI_Waitall( 2*10   , req, MPI_STATUSES_IGNORE );

            double t1 = MPI_Wtime();

            dt[links] = t1-t0;
            MPI_Allreduce( MPI_IN_PLACE, &dt[links], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
            dt[links] /= world_size;

            /* check for correctness */
            int error = 0;
            for ( int i = 0 ; i < count ; i++) error += abs( tag_ap - rbuf_ap[i] );

            if (error>0) printf("%d: %d errors \n", world_rank, error );
            assert(error==0);

            if (world_rank==0 && links>0) 
            {
                printf("%d: send %10d bytes along %2d links in %12.6lf seconds (%12.6lf MB/s/link/rpn, %12.6lf MB/s/rpn, %12.6lf MB/s) \n",
                       world_rank, (int) sizeof(int)*count, links, dt[links], 
                       1e-6 * count * sizeof(int) / dt[links],
                       1e-6 * links * count * sizeof(int) / dt[links],
                       1e-6 * links * node_size * count * sizeof(int) / dt[links]);
            }
        }
        fflush( stdout );
    }

    free(rbuf_ap);
    free(rbuf_am);
    free(rbuf_bp);
    free(rbuf_bm);
    free(rbuf_cp);
    free(rbuf_cm);
    free(rbuf_dp);
    free(rbuf_dm);
    free(rbuf_ep);
    free(rbuf_em);

    free(sbuf_ap);
    free(sbuf_am);
    free(sbuf_bp);
    free(sbuf_bm);
    free(sbuf_cp);
    free(sbuf_cm);
    free(sbuf_dp);
    free(sbuf_dm);
    free(sbuf_ep);
    free(sbuf_em);

    MPI_Barrier( MPI_COMM_WORLD );

    if (world_rank==0) printf("#done with all tests\n");
    fflush( stdout );

    MPI_Finalize();

    return 0;
}
