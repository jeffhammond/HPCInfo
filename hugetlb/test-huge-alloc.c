#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "huge-alloc.h"

int main(int argc, char* argv[])
{
    long long n = (argc>1) ? atoll(argv[1]) : 1LL<<21;
    size_t bytes = n * sizeof(uint64_t);
    size_t pagesize = (bytes >= 1L<<21) ? 1UL<<30 : 1UL<<21;
    uint64_t * data = huge_alloc(bytes,pagesize);
    for (size_t i=0; i<n; i++) {
        printf("%zu\n",i);
        data[i] = (uint64_t)i;
    }
    huge_free(data,bytes);
    return 0;
}
