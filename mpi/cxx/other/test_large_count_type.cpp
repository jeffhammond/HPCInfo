#include "large_count_type.hpp"
#include <cstring> // memset

int main(int argc, char* argv[])
{
    MPI_Init(&argc,&argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = (argc>1) ? atol(argv[1]) : 1000;

    char * buf = NULL;
    MPI_Alloc_mem(n, MPI_INFO_NULL, &buf);
    if (rank==0) {
        memset(buf, 'Z', n);
    } else {
        memset(buf, 'A', n);
    }

    {
        MpiTypeWrapper bigtype(n,MPI_CHAR);
        MPI_Bcast(buf,
                  bigtype.GetMpiCount(), bigtype.GetMpiDatatype(),
                  0, MPI_COMM_WORLD);
        /* ~MpiTypeWrapper happens here */
    }

    void * test = malloc(n);
    memset(test, 'Z', n);
    int rc = memcmp(static_cast<void*>(buf), test, n);
    if (rc!=0) {
        std::cout << "There were " << n << " error!" << std::endl;
    }

    free(test);
    MPI_Free_mem(buf);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
