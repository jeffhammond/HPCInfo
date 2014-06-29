#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>
#include <mpi.h>

/* this is to ensure that the threads overlap in time */
#define NAPTIME 3

#define MAX_POSIX_THREADS 64

static pthread_t thread_pool[MAX_POSIX_THREADS];

static int mpi_size, mpi_rank;
static int num_posix_threads;

void* foo(void* dummy)
{
    int i, my_pth = -1;
    pthread_t my_pthread = pthread_self();

    for (i=0 ; i<num_posix_threads ; i++)
        if (my_pthread==thread_pool[i]) my_pth = i;
    
    sleep(NAPTIME);

    int my_core = -1, my_hwth = -1;
    int my_omp, num_omp;
#pragma omp parallel private(my_core,my_hwth,my_omp,num_omp) shared(my_pth)
    {
        sleep(NAPTIME);

        my_core = get_bgq_core();
        my_hwth = get_bgq_hwthread();

        my_omp  = omp_get_thread_num();
        num_omp = omp_get_num_threads();
        fprintf(stdout,"MPI rank = %2d Pthread = %2d OpenMP thread = %2d of %2d core = %2d:%1d \n",
                       mpi_rank, my_pth, my_omp, num_omp, my_core, my_hwth);
        fflush(stdout);

        sleep(NAPTIME);
    }

    sleep(NAPTIME);

    pthread_exit(0);
}

void bar()
{
    sleep(NAPTIME);

    int my_core = -1, my_hwth = -1;
    int my_omp, num_omp;
#pragma omp parallel private(my_core,my_hwth,my_omp,num_omp)
    {
        sleep(NAPTIME);

        my_core = get_bgq_core();
        my_hwth = get_bgq_hwthread();

        my_omp  = omp_get_thread_num();
        num_omp = omp_get_num_threads();
        fprintf(stdout,"MPI rank = %2d OpenMP thread = %2d of %2d core = %2d:%1d \n",
                       mpi_rank, my_omp, num_omp, my_core, my_hwth);
        fflush(stdout);

        sleep(NAPTIME);
    }
    sleep(NAPTIME);
}

int main(int argc, char *argv[])
{
    int i, rc;
    int provided;
 
    MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
    if ( provided != MPI_THREAD_MULTIPLE ) exit(1);
 
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
 
    MPI_Barrier(MPI_COMM_WORLD);
 
    sleep(NAPTIME);

#ifdef __bgq__
    int bg_threadlayout = atoi(getenv("BG_THREADLAYOUT"));
    if (mpi_rank==0) fprintf(stdout,"BG_THREADLAYOUT = %2d\n", bg_threadlayout);
#endif

    num_posix_threads = atoi(getenv("POSIX_NUM_THREADS"));
    if (num_posix_threads<0)                 num_posix_threads = 0;
    if (num_posix_threads>MAX_POSIX_THREADS) num_posix_threads = MAX_POSIX_THREADS;

    if (mpi_rank==0) fprintf(stdout,"POSIX_NUM_THREADS = %2d\n", num_posix_threads);
    if (mpi_rank==0) fprintf(stdout,"OMP_MAX_NUM_THREADS = %2d\n", omp_get_max_threads());
    fflush(stdout);

    if ( num_posix_threads > 0 ) {
        //fprintf(stdout,"MPI rank %2d creating %2d POSIX threads\n", mpi_rank, num_posix_threads); fflush(stdout);
        for (i=0 ; i<num_posix_threads ; i++){
            rc = pthread_create(&thread_pool[i], NULL, foo, NULL);
            assert(rc==0);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        sleep(NAPTIME);

        for (i=0 ; i<num_posix_threads ; i++){
            rc = pthread_join(thread_pool[i],NULL);
            assert(rc==0);
        }
        //fprintf(stdout,"MPI rank %2d joined %2d POSIX threads\n", mpi_rank, num_posix_threads); fflush(stdout);
    } else {
        bar();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    sleep(NAPTIME);

    MPI_Finalize();

    return 0;
}
