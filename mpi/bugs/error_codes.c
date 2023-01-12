#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int main(void)
{
    MPI_Init(NULL,NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char filename[] = "nonexistent";

    int rc = MPI_File_delete(filename, MPI_INFO_NULL);

    int len;
    char errname[MPI_MAX_ERROR_STRING] = {0};

    MPI_Error_string(rc, errname, &len);
    printf("FILE err=%d, len=%d, name=%s\n", rc, len, errname);

    int class;
    MPI_Error_class(rc, &class);

    MPI_Error_string(class, errname, &len);
    printf("FILE err=%d, len=%d, name=%s\n", class, len, errname);

    int lc = MPI_ERR_LASTCODE;
    MPI_Error_string(lc, errname, &len);
    printf("LAST err=%d, len=%d, name=%s\n", lc, len, errname);

    MPI_Finalize();

    return 0;
}
