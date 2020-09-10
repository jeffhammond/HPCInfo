#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <mpi.h>

static inline
bool string_match(char * string, char * substring)
{
    // this should never happen...
    if (0 == strlen(substring)) return false;

    char * pos = strstr(string, substring);
    return (pos != NULL);
}

int main(void)
{
  MPI_Init(NULL,NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  {
      int version = 0;
      int subversion = 0;
      MPI_Get_version(&version, &subversion);

      if ((version != MPI_VERSION) || (subversion != MPI_SUBVERSION)) {
          printf("The MPI (version,subversion) is not consistent:\n compiled: (%d,%d)\n runtime:  (%d,%d)\n",
                  MPI_VERSION, MPI_SUBVERSION, version, subversion);
      } else {
          printf("The MPI (version,subversion) is consistent:\n compiled: (%d,%d)\n runtime:  (%d,%d)\n",
                  MPI_VERSION, MPI_SUBVERSION, version, subversion);
      }
  }

  {
      int resultlen = 0;
      char version[MPI_MAX_LIBRARY_VERSION_STRING+1] = {0};
      MPI_Get_library_version(version, &resultlen);
      printf("MPI_Get_library_version = %s\n", version);

#if defined(OPEN_MPI)
      bool mpich = string_match(version, "MPICH");
      if (mpich) {
          printf("Program was compiled with Open-MPI, but runtime library is MPICH-based - this will not work!\n");
      }
#endif
#if defined(MPICH) || defined(MPICH_VERSION) || defined(I_MPI_VERSION)
      bool ompi  = string_match(version, "Open");
      if (ompi) {
          printf("Program was compiled with Intel MPI, but runtime library is Open-MPI - this will not work!\n");
      }
#endif

  }

  MPI_Finalize();

  return 0;
}
