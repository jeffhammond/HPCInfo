#include <stdio.h>
#include <stdint.h>

int MPIR_F08_MPI_IN_PLACE;

void p(void)
{
    printf("MPIR_F08_MPI_IN_PLACE=%d &MPIR_F08_MPI_IN_PLACE=%p &MPIR_F08_MPI_IN_PLACE=%zu\n",
            MPIR_F08_MPI_IN_PLACE,   &MPIR_F08_MPI_IN_PLACE,   (intptr_t)&MPIR_F08_MPI_IN_PLACE);
}

void MPI_Allreduce(void ** sendbuf, void ** recvbuf,
                   int * count, int * datatype,
                   int * op, int * comm, int * ierror)
{
    printf("sendbuf=%p, sendbuf=%zu\n", sendbuf, (intptr_t)sendbuf);
    printf("sendbuf is MPI_IN_PLACE? %s\n", 
           (intptr_t)sendbuf==(intptr_t)&MPIR_F08_MPI_IN_PLACE ? "yes" : "no");
    printf("recvbuf=%p, recvbuf=%zu\n", recvbuf, (intptr_t)recvbuf);
    printf("*count=%d, *datatype=%d, *op=%d, *comm=%d\n",
            *count, *datatype, *op, *comm);
    *ierror = 911;
}
