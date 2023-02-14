#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <mpi.h>

// int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)

typedef struct {
    const void * sbuf; 
    void * rbuf;
    int count;
    MPI_Datatype dt;
    MPI_Op op;
    int root;
    MPI_Comm comm;
    MPI_Request req;
} reduce_args;

int query_fn(void * extra_state, MPI_Status *status)
{
    printf("query_fn called\n"); 
    reduce_args * args = (reduce_args*)extra_state;
    MPI_Status_set_elements(status, args->dt, args->count);
    MPI_Status_set_cancelled(status, 0);
    status->MPI_SOURCE = MPI_UNDEFINED;
    status->MPI_TAG = MPI_UNDEFINED;
    return MPI_SUCCESS;
}

int free_fn(void * extra_state)
{
    printf("free_fn called\n"); 
    return MPI_SUCCESS;
}

int cancel_fn(void * extra_state, int complete)
{
    printf("cancel_fn called\n"); 
    if (!complete) MPI_Abort(MPI_COMM_WORLD, complete);
    return MPI_SUCCESS;
}

void * reduce_fn(void * ptr)
{
    printf("reduce_fn called\n"); 
    reduce_args * args = (reduce_args*)ptr;
    MPI_Reduce(args->sbuf, args->rbuf, args->count, args->dt, args->op, args->root, args->comm);
    MPI_Grequest_complete(args->req);
    return NULL;
}

int My_Ireduce(const void * sendbuf, void * recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
               MPI_Request * request)
{
    pthread_t thread;

    reduce_args * args = malloc(sizeof(reduce_args));

    MPI_Grequest_start(query_fn, free_fn, cancel_fn, args, request);

    args -> sbuf  = sendbuf;
    args -> rbuf  = recvbuf;
    args -> count = count;
    args -> dt    = datatype; // technically, datatype can be freed before this is used, so it should be dup'd
    args -> op    = op;       // probably the same as datatype
    args -> root  = root;
    args -> comm  = comm;
    args -> req   = *request;

    int rc = pthread_create(&thread, NULL /* attr */, reduce_fn, args);
    if (rc) {
        printf("pthread_create returned %d\n", rc);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    return MPI_SUCCESS;
}


int main(int argc, char * argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) MPI_Abort(MPI_COMM_WORLD, provided);

    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int count = 100;
    double * x = malloc( count * sizeof(double) );
    double * y = malloc( count * sizeof(double) );
    for (int i=0; i<count; i++) {
        x[i] = i;
        y[i] = 0;
    }
    
    int root = 0;
    MPI_Request r;
    My_Ireduce(x, y, count, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD, &r);

    MPI_Status s;
    MPI_Wait(&r, &s);

    int out_count;
    MPI_Get_count(&s, MPI_DOUBLE, &out_count);
    printf("%d: out count = %d\n", me, out_count);

    if (me == root) {
        for (int i=0; i<count; i++) {
            if (y[i] != np * x[i]) {
                printf("error: x[%d]=%f y[%d]=%f ref=%f\n", i, x[i], i, y[i], np * x[i]);
            }
        }
    }

    free(x);
    free(y);

    MPI_Finalize();

    return 0;
}
