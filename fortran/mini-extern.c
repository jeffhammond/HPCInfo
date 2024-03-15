#include <stdio.h>
#include <stdint.h>

int MPIR_F08_MPI_IN_PLACE;

void p(void)
{
    printf("MPIR_F08_MPI_IN_PLACE=%d &MPIR_F08_MPI_IN_PLACE=%p &MPIR_F08_MPI_IN_PLACE=%zu\n",
            MPIR_F08_MPI_IN_PLACE,   &MPIR_F08_MPI_IN_PLACE,   (intptr_t)&MPIR_F08_MPI_IN_PLACE);
}
