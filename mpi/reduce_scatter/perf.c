#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

// we don't bother with Reduce-then-Scatterv because it requires a displacement vector

typedef enum {
    MPI_REDUCE_SCATTER = 0,
    MPI_REDUCE_SCATTER_BLOCK = 1,
    REDUCE_THEN_SCATTER = 2,
    ALLREDUCE_MEMCPY = 3,
    MULTIROOT_REDUCE = 4,
    MULTIROOT_IREDUCE = 5,
    VARIANT_MAX = 6
} variant_e;

void print_variant_name(variant_e e)
{
    switch (e)
    {
        case MPI_REDUCE_SCATTER:
        {
            printf("MPI_REDUCE_SCATTER");
            break;
        }
        case MPI_REDUCE_SCATTER_BLOCK:
        {
            printf("MPI_REDUCE_SCATTER_BLOCK");
            break;
        }
        case REDUCE_THEN_SCATTER:
        {
            printf("REDUCE_THEN_SCATTER");
            break;
        }
        case ALLREDUCE_MEMCPY:
        {
            printf("ALLREDUCE_MEMCPY");
            break;
        }
        case MULTIROOT_REDUCE:
        {
            printf("MULTIROOT_REDUCE");
            break;
        }
        case MULTIROOT_IREDUCE:
        {
            printf("MULTIROOT_IREDUCE");
            break;
        }
        default:
        {
            abort();
        }
    }
}

int test_reduce_scatter(void * input, void * output, int count, int * counts,
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
            rc = MPI_Reduce(MPI_IN_PLACE, input, np * count, type, op, root, comm);
            rc = MPI_Scatter(input, count, type, output, count, type, root, comm);
            break;
        }
        case ALLREDUCE_MEMCPY:
        {
            rc = MPI_Allreduce(MPI_IN_PLACE, input, np * count, type, op, comm);
            memcpy(output, &input[me * count], count);
            break;
        }
        case MULTIROOT_REDUCE:
        {
            for (int r=0; r<np; r++) {
                rc = MPI_Reduce(&input[me * count], output, count, type, op, r, comm);
            }
            break;
        }
        case MULTIROOT_IREDUCE:
        {
            for (int r=0; r<np; r++) {
                rc = MPI_Ireduce(&input[me * count], output, count, type, op, r, comm, &reqs[r]);
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
        printf("reduce scatter test: count = %d\n", count);
    }
    rc = MPI_Bcast(&count, 1, MPI_INT, 0, comm);

    MPI_Op op = MPI_SUM;
    MPI_Datatype type = MPI_INT64_T;
    int64_t * input  = calloc(count*np,sizeof(int64_t));
    int64_t * output = calloc(count,sizeof(int64_t));

    if (me==0) {
        for (int r=0; r<np; r++) {
            for (int k=0; k<count; k++) {
                input[r*(size_t)count + k] = 1;
            }
        }
    }

    int * counts = malloc(np*sizeof(int));
    for (int r=0; r<np; r++) {
        counts[r] = count;
    }

    double t0, t1;
    for (int e=0; e<VARIANT_MAX; e++) {
        for (int j=0; j<REPS; j++) {
            if (me==0 && j==0) {
                printf("variant %d:", e);
                print_variant_name(e);
                printf("\n");
            }
            else if (j==1) {
                rc = MPI_Barrier(comm);
                t0 = MPI_Wtime();
            }

            rc = test_reduce_scatter(input, output, count, counts, type, op, comm, e, me, np, reqs);
            if (rc != MPI_SUCCESS) MPI_Abort(comm,rc);

            if (j==0) {
                // verification
            }
        }
        rc = MPI_Barrier(comm);
        t1 = MPI_Wtime();
        if (me==0) {
            printf("dt = %f\n", t1-t0);
        }
    }

    free(input);
    free(output);
    free(reqs);

    MPI_Finalize();

    return 0;
}
