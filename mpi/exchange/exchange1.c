#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>

int posix_memalign(void **memptr, size_t alignment, size_t size);

#define ALIGNMENT 64

#ifdef 
#  include <mpix.h>
#else
#  warning This test should be run on BGP.
#endif

int main(int argc, char *argv[])
{
    int provided;
    int world_size, world_rank;

    MPI_Init_thread( &argc, &argv, MPI_THREAD_SINGLE, &provided );
    assert( provided==MPI_THREAD_SINGLE );

    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
#ifdef 
    assert(world_size>=27);
#else
    assert(world_size>=2);
#endif

    int max_links = ( argc > 1 ? atoi(argv[1]) : world_size-1 );
    int max_count = ( argc > 2 ? atoi(argv[2]) : 16*1024*1024 );
    int nbrecv    = ( argc > 3 ? atoi(argv[3]) : 0 );

#ifdef 
    uint32_t xSize, ySize, zSize, tSize;
    MPIX_rank2torus( world_size-1, &xSize, &ySize, &zSize, &tSize );
    if (world_rank==0) printf("#torus size = (%d,%d,%d) \n", xSize+1, ySize+1, zSize+1 );

    uint32_t xRank, yRank, zRank, tRank;
    MPIX_rank2torus( world_rank, &xRank, &yRank, &zRank, &tRank );
    //printf("I am %d (%d,%d,%d). \n", world_rank, xRank, yRank, zRank );

    int rank_c0 = MPIX_torus2rank(1,1,1,0);
    int rank_xp = MPIX_torus2rank(2,1,1,0);
    int rank_xm = MPIX_torus2rank(0,1,1,0);
    int rank_yp = MPIX_torus2rank(1,2,1,0);
    int rank_ym = MPIX_torus2rank(1,0,1,0);
    int rank_zp = MPIX_torus2rank(1,1,2,0);
    int rank_zm = MPIX_torus2rank(1,1,0,0);

    if (world_rank==0) printf("#send from %d (1,1,1) to %d (2,1,1), %d (0,1,1), %d (1,2,1), %d (1,0,1), %d (1,1,2), %d (1,1,0) \n", 
                              rank_c0, rank_xp, rank_xm, rank_yp, rank_ym, rank_zp, rank_zm );
#else
    int rank_c0 = 0;
    int rank_xp = 1;
    int rank_xm = 2;
    int rank_yp = 3;
    int rank_ym = 4;
    int rank_zp = 5;
    int rank_zm = 6;

    if (world_rank==0) printf("#send from %d to %d, %d, %d, %d, %d, %d \n", 
                              rank_c0, rank_xp, rank_xm, rank_yp, rank_ym, rank_zp, rank_zm );
#endif

    int * rbuf_xp = malloc((size_t) max_count * sizeof(int));
    int * rbuf_xm = malloc((size_t) max_count * sizeof(int));
    int * rbuf_yp = malloc((size_t) max_count * sizeof(int));
    int * rbuf_ym = malloc((size_t) max_count * sizeof(int));
    int * rbuf_zp = malloc((size_t) max_count * sizeof(int));
    int * rbuf_zm = malloc((size_t) max_count * sizeof(int));
    
    assert( rbuf_xp != NULL && rbuf_xm != NULL && rbuf_yp != NULL && rbuf_ym != NULL && rbuf_zp != NULL && rbuf_zm != NULL);
    
    int * sbuf_xp = malloc((size_t) max_count * sizeof(int));
    int * sbuf_xm = malloc((size_t) max_count * sizeof(int));
    int * sbuf_yp = malloc((size_t) max_count * sizeof(int));
    int * sbuf_ym = malloc((size_t) max_count * sizeof(int));
    int * sbuf_zp = malloc((size_t) max_count * sizeof(int));
    int * sbuf_zm = malloc((size_t) max_count * sizeof(int));
    
    assert( sbuf_xp != NULL && sbuf_xm != NULL && sbuf_yp != NULL && sbuf_ym != NULL && sbuf_zp != NULL && sbuf_zm != NULL);

    if (world_rank==0) printf("#begin nonblocking send-recv 3d halo exchange test\n" );

    fflush( stdout );
    MPI_Barrier( MPI_COMM_WORLD );

    for ( int count = 1; count <= max_count ; count *= 2 )
    {
        double dt[6] = {1.0e9,1.0e9,1.0e9,1.0e9,1.0e9,1.0e9};

        int rc[7] = {0,0,0,0,0,0,0};
        MPI_Request req[6];

        int tag_xp = 6*count;
        int tag_xm = 6*count+1;
        int tag_yp = 6*count+2;
        int tag_ym = 6*count+3;
        int tag_zp = 6*count+4;
        int tag_zm = 6*count+5;
     
        /* links = 0 is warmup on all 6 links */
        for ( int links = 0; links <= max_links ; links++ )
        { 
            if (links==0 || links>0)
            {
                for ( int i = 0 ; i < count ; i++) rbuf_xp[i] = 0;
                for ( int i = 0 ; i < count ; i++) sbuf_xp[i] = 6*i;
            }
            if (links==0 || links>1)
            {
                for ( int i = 0 ; i < count ; i++) rbuf_xm[i] = 0;
                for ( int i = 0 ; i < count ; i++) sbuf_xm[i] = 6*i+1;
            }
            if (links==0 || links>2)
            {
                for ( int i = 0 ; i < count ; i++) rbuf_yp[i] = 0;
                for ( int i = 0 ; i < count ; i++) sbuf_yp[i] = 6*i+2;
            }
            if (links==0 || links>3)
            {
                for ( int i = 0 ; i < count ; i++) rbuf_ym[i] = 0;
                for ( int i = 0 ; i < count ; i++) sbuf_ym[i] = 6*i+3;
            }
            if (links==0 || links>4)
            {
                for ( int i = 0 ; i < count ; i++) rbuf_zp[i] = 0;
                for ( int i = 0 ; i < count ; i++) sbuf_zp[i] = 6*i+4;
            }
            if (links==0 || links>5)
            {
                for ( int i = 0 ; i < count ; i++) rbuf_zm[i] = 0;
                for ( int i = 0 ; i < count ; i++) sbuf_zm[i] = 6*i+5;
            }

            MPI_Barrier( MPI_COMM_WORLD );

            double t0 = MPI_Wtime();

            if (links==0 || links>0)
            {
                if ( world_rank==rank_xp )
                {
                    if (nbrecv==1)
                    {
                        rc[0] = MPI_Irecv( rbuf_xp, count, MPI_INT, rank_c0, tag_xp, MPI_COMM_WORLD, &req[0] );
                        sleep(1);
                        rc[1] = MPI_Wait( &req[0], MPI_STATUSES_IGNORE ); 
                    }
                    else
                        rc[0] = MPI_Recv( rbuf_xp, count, MPI_INT, rank_c0, tag_xp, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                }

                if ( world_rank==rank_c0 )
                    rc[1] = MPI_Isend( sbuf_xp, count, MPI_INT, rank_xp, tag_xp, MPI_COMM_WORLD, &req[0] );
            }
            if (links==0 || links>1)
            {
                if ( world_rank==rank_xm )
                {
                    if (nbrecv==1)
                    {
                        rc[0] = MPI_Irecv( rbuf_xm, count, MPI_INT, rank_c0, tag_xm, MPI_COMM_WORLD, &req[1] );
                        sleep(1);
                        rc[1] = MPI_Wait( &req[1], MPI_STATUSES_IGNORE ); 
                    }
                    else
                        rc[0] = MPI_Recv( rbuf_xm, count, MPI_INT, rank_c0, tag_xm, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                }

                if ( world_rank==rank_c0 )
                    rc[2] = MPI_Isend( sbuf_xm, count, MPI_INT, rank_xm, tag_xm, MPI_COMM_WORLD, &req[1] );
            }
            if (links==0 || links>2)
            {
                if ( world_rank==rank_yp )
                {
                    if (nbrecv==1)
                    {
                        rc[0] = MPI_Irecv( rbuf_yp, count, MPI_INT, rank_c0, tag_yp, MPI_COMM_WORLD, &req[2] );
                        sleep(1);
                        rc[1] = MPI_Wait( &req[2], MPI_STATUSES_IGNORE ); 
                    }
                    else
                        rc[0] = MPI_Recv( rbuf_yp, count, MPI_INT, rank_c0, tag_yp, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                }

                if ( world_rank==rank_c0 )
                    rc[3] = MPI_Isend( sbuf_yp, count, MPI_INT, rank_yp, tag_yp, MPI_COMM_WORLD, &req[2] );
            }
            if (links==0 || links>3)
            {
                if ( world_rank==rank_ym )
                {
                    if (nbrecv==1)
                    {
                        rc[0] = MPI_Irecv( rbuf_ym, count, MPI_INT, rank_c0, tag_ym, MPI_COMM_WORLD, &req[3] );
                        sleep(1);
                        rc[1] = MPI_Wait( &req[3], MPI_STATUSES_IGNORE ); 
                    }
                    else
                        rc[0] = MPI_Recv( rbuf_ym, count, MPI_INT, rank_c0, tag_ym, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                }

                if ( world_rank==rank_c0 )
                    rc[4] = MPI_Isend( sbuf_ym, count, MPI_INT, rank_ym, tag_ym, MPI_COMM_WORLD, &req[3] );
            }
            if (links==0 || links>4)
            {
                if ( world_rank==rank_zp )
                {
                    if (nbrecv==1)
                    {
                        rc[0] = MPI_Irecv( rbuf_zp, count, MPI_INT, rank_c0, tag_zp, MPI_COMM_WORLD, &req[4] );
                        sleep(1);
                        rc[1] = MPI_Wait( &req[4], MPI_STATUSES_IGNORE ); 
                    }
                    else
                        rc[0] = MPI_Recv( rbuf_zp, count, MPI_INT, rank_c0, tag_zp, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                }

                if ( world_rank==rank_c0 )
                    rc[5] = MPI_Isend( sbuf_zp, count, MPI_INT, rank_zp, tag_zp, MPI_COMM_WORLD, &req[4] );
            }
            if (links==0 || links>5)
            {
                if ( world_rank==rank_zm )
                {
                    if (nbrecv==1)
                    {
                        rc[0] = MPI_Irecv( rbuf_zm, count, MPI_INT, rank_c0, tag_zm, MPI_COMM_WORLD, &req[5] );
                        sleep(1);
                        rc[1] = MPI_Wait( &req[5], MPI_STATUSES_IGNORE ); 
                    }
                    else
                        rc[0] = MPI_Recv( rbuf_zm, count, MPI_INT, rank_c0, tag_zm, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                }

                if ( world_rank==rank_c0 )
                    rc[6] = MPI_Isend( sbuf_zm, count, MPI_INT, rank_zm, tag_zm, MPI_COMM_WORLD, &req[5] );
            }

            if ( world_rank==rank_c0 )
                rc[0] = MPI_Waitall( links, req, MPI_STATUSES_IGNORE ); 

            double t1 = MPI_Wtime();

            MPI_Barrier( MPI_COMM_WORLD );

            if ( world_rank==rank_c0 )
            {
                assert( rc[0]==MPI_SUCCESS );
                for ( int i = 1 ; i <= links ; i++ ) 
                    assert( rc[i]==MPI_SUCCESS );
            }
            else if ( world_rank==rank_xp || world_rank==rank_xm || world_rank==rank_yp || world_rank==rank_ym || world_rank==rank_zp || world_rank==rank_zm )
            {
                assert( rc[0]==MPI_SUCCESS );
                if (nbrecv==1) 
                    assert( rc[1]==MPI_SUCCESS );
            }

            /* check for correctness */
            int error = 0;

            if (links>0)
                if ( world_rank==rank_xp )
                    for ( int i = 0 ; i < count ; i++) 
                        error += abs( 6*i   - rbuf_xp[i] );

            if (links>1)
                if ( world_rank==rank_xm )
                    for ( int i = 0 ; i < count ; i++) 
                        error += abs( 6*i+1 - rbuf_xm[i] );

            if (links>2)
                if ( world_rank==rank_yp )
                    for ( int i = 0 ; i < count ; i++) 
                        error += abs( 6*i+2 - rbuf_yp[i] );

            if (links>3)
                if ( world_rank==rank_ym )
                    for ( int i = 0 ; i < count ; i++) 
                        error += abs( 6*i+3 - rbuf_ym[i] );

            if (links>4)
                if ( world_rank==rank_zp )
                    for ( int i = 0 ; i < count ; i++) 
                        error += abs( 6*i+4 - rbuf_zp[i] );

            if (links>5)
                if ( world_rank==rank_zm )
                    for ( int i = 0 ; i < count ; i++) 
                        error += abs( 6*i+5 - rbuf_zm[i] );

            if (error>0) printf("%d: %d errors \n", world_rank, error );
            assert(error==0);
     
            if (links>0) dt[links-1] = t1-t0;
        }
        if (world_rank==rank_c0) printf("%d: send %10d bytes %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf MB/s (1, 2, 3, 4, 5, 6 links)\n", 
                                        world_rank, (int) sizeof(int)*count, 
                                        1e-6*1*count*sizeof(int)/dt[0], 1e-6*2*count*sizeof(int)/dt[1],
                                        1e-6*3*count*sizeof(int)/dt[2], 1e-6*4*count*sizeof(int)/dt[3],
                                        1e-6*5*count*sizeof(int)/dt[4], 1e-6*6*count*sizeof(int)/dt[5]);
        fflush( stdout );
    }

    free(rbuf_xp);
    free(rbuf_xm);
    free(rbuf_yp);
    free(rbuf_ym);
    free(rbuf_zp);
    free(rbuf_zm);
           
    free(sbuf_xp);
    free(sbuf_xm);
    free(sbuf_yp);
    free(sbuf_ym);
    free(sbuf_zp);
    free(sbuf_zm);

    MPI_Barrier( MPI_COMM_WORLD );

    if (world_rank==0) printf("#done with all tests\n");
    fflush( stdout );

    MPI_Finalize();

    return 0;
}

