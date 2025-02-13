#include <stdio.h>
#include <mpi.h>

void intercept_(MPI_Status * s)
{
    printf("&MPI_STATUS_IGNORE = %p\n",s);
    printf("&MPI_STATUS_IGNORE = %zu\n",(intptr_t)s);
}
