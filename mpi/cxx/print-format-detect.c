#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int main(void)
{
    MPI_Count  c = 5;
    MPI_Aint   a = 6;
    MPI_Offset o = 7;

    switch (sizeof(c)) {
        case 4:
            if (c<0) printf("%d\n",c);
            else     printf("%u\n",c);
            break;
        case 8:
            if (c<0) printf("%lld\n",c);
            else     printf("%llu\n",c);
            break;
        default: abort(); break;
    }
    return 0;
}
