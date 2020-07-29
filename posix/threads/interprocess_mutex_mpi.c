#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <pthread.h>

int main(int argc, char* argv[])
{
    int rc;  /* for MPI   */
    int err; /* for POSIX */

    int provided;
    rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);

    MPI_Comm node_comm;
    rc = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);

    int node_rank;
    MPI_Comm_rank(node_comm, &node_rank);

    pthread_mutex_t * shm_mutex;

    MPI_Win shm_mutex_win;
    rc = MPI_Win_allocate_shared(node_rank==0 ? sizeof(pthread_mutex_t) : 0,
                                 1, MPI_INFO_NULL, node_comm, &shm_mutex, &shm_mutex_win);
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);

    MPI_Aint size; /* unused */
    int disp;      /* unused */
    rc =  MPI_Win_shared_query(shm_mutex_win, 0, &size, &disp, &shm_mutex);
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);

    printf("node_rank=%d, shm_mutex=%p\n", node_rank, shm_mutex);

    if (node_rank==0) {
        pthread_mutexattr_t attr;

        err = pthread_mutexattr_init(&attr);
        if (err != 0) MPI_Abort(MPI_COMM_WORLD, rc);

        err = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        if (err != 0) MPI_Abort(MPI_COMM_WORLD, rc);

        err = pthread_mutex_init(shm_mutex, &attr);
        if (err != 0) MPI_Abort(MPI_COMM_WORLD, rc);

        err = pthread_mutexattr_destroy(&attr);
        if (err != 0) MPI_Abort(MPI_COMM_WORLD, rc);
    }

    {
        err = pthread_mutex_lock(shm_mutex);
        if (err != 0) MPI_Abort(MPI_COMM_WORLD, rc);

        err = pthread_mutex_unlock(shm_mutex);
        if (err != 0) MPI_Abort(MPI_COMM_WORLD, rc);
    }

    if (node_rank==0) {
        err = pthread_mutex_destroy(shm_mutex);
        if (err != 0) MPI_Abort(MPI_COMM_WORLD, rc);
    }

    rc = MPI_Win_free(&shm_mutex_win);
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);

    rc = MPI_Comm_free(&node_comm);
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);

    rc = MPI_Finalize();
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);

    return 0;
}
