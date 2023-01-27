#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

// we don't bother with Reduce-then-Scatterv becaues it requires a displacement vector

typedef enum {
    MPI_REDUCE_SCATTER = 0,
    MPI_REDUCE_SCATTER_BLOCK = 1,
    REDUCE_THEN_SCATTER = 2,
    ALLREDUCE_MEMCPY = 3,
    MULTIROOT_REDUCE = 4,
    MULTIROOT_IREDUCE = 5
} variant_e;

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
            for (int i=0; i<np; i++) {
                rc = MPI_Reduce(&input[me * count], output, count, type, op, me, comm);
            }
            break;
        }
        case MULTIROOT_IREDUCE:
        {
            for (int i=0; i<np; i++) {
                rc = MPI_Ireduce(&input[me * count], output, count, type, op, me, comm, &reqs[i]);
            }
            rc = MPI_Waitall(count, reqs, MPI_STATUSES_IGNORE);
            break;
        }
    }
    return rc;
}


int main(int argc, char* argv[])
{
    MPI_Comm comm = MPI_COMM_WORLD;

    int me, np;
    MPI_Comm_size(comm, &me);
    MPI_Comm_size(comm, &np);

    MPI_Request * reqs = malloc(np * sizeof(MPI_Request));

#if 0
    for (int i=0; 
        test_reduce_scatter(void * input, void * output, int count, int * counts,
                            MPI_Datatype type, MPI_Op op, MPI_Comm comm,
                            variant_e e, int me, int np, MPI_Request reqs[])
#endif

    free(reqs);

    return 0;
}
