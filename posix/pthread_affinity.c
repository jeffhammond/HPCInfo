// this has to be above stdio.h for sched_getaffinity to be declared
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <sched.h>
#include <unistd.h>

#include <pthread.h>

#include <mpi.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))

const useconds_t naptime = 1000;

static void * nagot(void * input)
{
    (void)input;

    cpu_set_t mask;
    CPU_ZERO(&mask);
    int rc = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
    assert(rc == 0);

    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (me > 0) MPI_Recv(NULL,0,MPI_CHAR,me-1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    fflush(stdout); usleep(naptime);
    printf("%d: pthread CPU set=", me);
    for (size_t i=0; i<CPU_SETSIZE; i++) {
        const int on = CPU_ISSET(i, &mask);
        if (on) printf(" %zu", i);
    }
    printf("\n\n"); fflush(stdout); usleep(naptime);

    if (me < (np-1)) MPI_Ssend(NULL,0,MPI_CHAR,me+1,1,MPI_COMM_WORLD);

    pthread_exit(NULL);
    return NULL;
}

int main(void)
{
    int provided;
    MPI_Init_thread(NULL,NULL,MPI_THREAD_MULTIPLE,&provided);
    assert(provided == MPI_THREAD_MULTIPLE);

    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    cpu_set_t mask;
    CPU_ZERO(&mask);

    int rc = sched_getaffinity(getpid(), sizeof(cpu_set_t), &mask);
    assert(rc == 0);

    if (me > 0) MPI_Recv(NULL,0,MPI_CHAR,me-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    fflush(stdout); usleep(naptime);
    printf("%d: main CPU set=", me);
    for (size_t i=0; i<sizeof(cpu_set_t); i++) {
        const int on = CPU_ISSET(i, &mask);
        if (on) printf(" %zu", i);
    }
    printf("\nCPU_COUNT=%d",CPU_COUNT(&mask));
    printf("\n\n"); fflush(stdout); usleep(naptime);

    if (me < (np-1)) MPI_Ssend(NULL,0,MPI_CHAR,me+1,0,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    int max_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (me == 0) printf("total online CPUs = %d\n", max_cpus);

    cpu_set_t newmask;
    CPU_ZERO(&newmask);
    for (size_t i=0; i<MAX(max_cpus,CPU_SETSIZE); i++) {
        CPU_SET(i, &newmask);
    }

    pthread_attr_t attr;
    rc = pthread_attr_init(&attr);
    assert(rc == 0);

    rc = pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &newmask);
    assert(rc == 0);

    pthread_t thread;
    rc = pthread_create(&thread, &attr, &nagot, NULL);
    assert(rc==0);

    rc = pthread_join(thread, NULL);
    assert(rc==0);

    rc = pthread_attr_destroy(&attr);
    assert(rc == 0);

    MPI_Finalize();

    return 0;
}
