#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>

#include <mpi.h>

// we don't bother with Reduce-then-Scatterv because it requires a displacement vector

typedef enum {
    MPI_REDUCE_SCATTER          = 0,
    MPI_REDUCE_SCATTER_BLOCK    = 1,
    REDUCE_THEN_SCATTER         = 2,
    REDUCE_THEN_SCATTER_OOP     = 3,
    ALLREDUCE_MEMCPY            = 4,
    ALLREDUCE_MEMCPY_OOP        = 5,
    MULTIROOT_REDUCE            = 6,
    MULTIROOT_IREDUCE           = 7,
    VARIANT_MAX                 = 8
} variant_e;

void print_variant_name(variant_e e)
{
    switch (e)
    {
        case MPI_REDUCE_SCATTER:
        {
            printf("%30s","MPI_REDUCE_SCATTER");
            break;
        }
        case MPI_REDUCE_SCATTER_BLOCK:
        {
            printf("%30s","MPI_REDUCE_SCATTER_BLOCK");
            break;
        }
        case REDUCE_THEN_SCATTER:
        {
            printf("%30s","REDUCE_THEN_SCATTER");
            break;
        }
        case REDUCE_THEN_SCATTER_OOP:
        {
            printf("%30s","REDUCE_THEN_SCATTER_OOP");
            break;
        }
        case ALLREDUCE_MEMCPY:
        {
            printf("%30s","ALLREDUCE_MEMCPY");
            break;
        }
        case ALLREDUCE_MEMCPY_OOP:
        {
            printf("%30s","ALLREDUCE_MEMCPY_OOP");
            break;
        }
        case MULTIROOT_REDUCE:
        {
            printf("%30s","MULTIROOT_REDUCE");
            break;
        }
        case MULTIROOT_IREDUCE:
        {
            printf("%30s","MULTIROOT_IREDUCE");
            break;
        }
        default:
        {
            printf("%30s","INVALID VARIANT");
            abort();
            break;
        }
    }
}

int test_reduce_scatter(int64_t * input, int64_t * output, int count, int * counts,
                        MPI_Datatype type, MPI_Op op, MPI_Comm comm,
                        variant_e e, int me, int np, MPI_Request reqs[])
{
    int rc = MPI_SUCCESS;

    switch (e)
    {
        case MPI_REDUCE_SCATTER:
        {
            rc = MPI_Reduce_scatter(input, output, counts, type, op, comm);
            break;
        }
        case MPI_REDUCE_SCATTER_BLOCK:
        {
            rc = MPI_Reduce_scatter_block(input, output, count, type, op, comm);
            break;
        }
        case REDUCE_THEN_SCATTER:
        {
            const int root = 0;
            rc = MPI_Reduce((me==root) ? MPI_IN_PLACE : input, input, np * count, type, op, root, comm);
            rc = MPI_Scatter(input, count, type, output, count, type, root, comm);
            break;
        }
        case REDUCE_THEN_SCATTER_OOP:
        {
            const int root = 0;
            int64_t * buf = (me==root) ? malloc(np * count * sizeof(int64_t)) : NULL;
            rc = MPI_Reduce(input, buf, np * count, type, op, root, comm);
            rc = MPI_Scatter(buf, count, type, output, count, type, root, comm);
            if (me==root) free(buf);
            break;
        }
        case ALLREDUCE_MEMCPY:
        {
            rc = MPI_Allreduce(MPI_IN_PLACE, input, np * count, type, op, comm);
            for (int k=0; k<count; k++) {            
                output[k] = input[me * count + k];
            }
            break;
        }
        case ALLREDUCE_MEMCPY_OOP:
        {
            int64_t * buf = malloc(np * count * sizeof(int64_t));
            rc = MPI_Allreduce(input, buf, np * count, type, op, comm);
            for (int k=0; k<count; k++) {
                output[k] = buf[me * count + k];
            }
            free(buf);
            break;
        }
        case MULTIROOT_REDUCE:
        {
            for (int r=0; r<np; r++) {
                rc = MPI_Reduce(&input[r * count], (r==me) ? output : NULL, count, type, op, r, comm);
            }
            break;
        }
        case MULTIROOT_IREDUCE:
        {
            for (int r=0; r<np; r++) {
                rc = MPI_Ireduce(&input[r * count], (r==me) ? output : NULL, count, type, op, r, comm, &reqs[r]);
            }
            rc = MPI_Waitall(np, reqs, MPI_STATUSES_IGNORE);
            break;
        }
        default:
        {
            rc = MPI_Abort(comm,e);
        }
    }
    return rc;
}

#define REPS 20

int main(int argc, char* argv[])
{
    int rc;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc,&argv);

    int me, np;
    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &np);

    MPI_Request * reqs = malloc(np * sizeof(MPI_Request));

    int count;
    if (me==0) {
        count = (argc > 1) ? atoi(argv[1]) : 1000;
        printf("reduce scatter test: count = %d (64-bit integers)\n", count);
    }
    rc = MPI_Bcast(&count, 1, MPI_INT, 0, comm);

    MPI_Op op = MPI_SUM;
    MPI_Datatype type = MPI_INT64_T;
    int64_t * input  = calloc(count*np,sizeof(int64_t));
    int64_t * output = calloc(count,sizeof(int64_t));
    int64_t * outref = calloc(count,sizeof(int64_t));

    int * counts = malloc(np*sizeof(int));
    for (int r=0; r<np; r++) {
        counts[r] = count;
    }

    double t0=0, t1=0;
    for (variant_e e=MPI_REDUCE_SCATTER; e!=VARIANT_MAX; e++)
    {
        // initialize buffers every time we change variants
        for (int r=0; r<np; r++) {
            for (int k=0; k<count; k++) {
                input[r*(size_t)count + k] = r * INT_MAX + k;
            }
        }
        memset(output, 0, count * sizeof(int64_t));

        // iteration 0: print and verify
        // subsequent iterations: timing
        for (int j=0; j<=REPS; j++) {
            if (me==0 && j==0) {
                printf("variant %d:", (int)e);
                print_variant_name(e);
                printf("\t");
            }
            else if (j==1) {
                rc = MPI_Barrier(comm);
                t0 = MPI_Wtime();
            }

            rc = test_reduce_scatter(input, output, count, counts, type, op, comm, e, me, np, reqs);
            if (rc != MPI_SUCCESS) MPI_Abort(comm,rc);

            if (j==0) {
                // verification
                if (e==0) {
                    memcpy(outref, output, count*sizeof(int64_t));
                } else {
                    int check = memcmp(outref, output, count*sizeof(int64_t));
                    if (check != 0) {
                        printf("variant %d rank %d: fails correctness check\n", e, me);
                        for (int k=0; k<count; k++) {
                            if (outref[k] != output[k]) {
                                printf("variant %d rank %d: output[%d]=%" PRId64 "\n", (int)e, me, k, output[k]);
                            }
                        }
                        fflush(stdout);
                    }
                }
            }
        }
        rc = MPI_Barrier(comm);
        t1 = MPI_Wtime();
        if (me==0) {
            printf("dt = %f seconds\n", (t1-t0) / REPS);
        }
    }

    free(input);
    free(output);
    free(reqs);

    MPI_Finalize();

    return 0;
}
