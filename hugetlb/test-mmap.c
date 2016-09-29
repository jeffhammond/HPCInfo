#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <sys/mman.h>
#include <errno.h>

int main(int argc, char* argv[])
{
    long long n = (argc>1) ? atoll(argv[1]) : 1024*1024*1024;
    for (size_t i=1; i<=n; i*=2) {
        size_t bytes = i;
        printf("bytes=%zu\n",bytes);
        {
            int prot  = PROT_READ | PROT_WRITE;      /* read-write memory    */
            int flags = MAP_ANONYMOUS |              /* no file backing      */ 
                        MAP_NORESERVE |              /* do not reserve swap  */
                        MAP_POPULATE |               /* pre-fault pages      */
                        MAP_PRIVATE;                 /* not shared memory    */
#if 0
                        MAP_LOCKED |                 /* lock/pin pages       */
                        MAP_HUGETLB |                /* huge pages           */
                        ;
#endif
            char * ptr = mmap(NULL, bytes, prot, flags, -1, 0);
            printf("ptr=%p\n", ptr);
            memset(ptr,0,bytes);
            int rc = munmap(ptr,bytes);
            printf("rc=%d\n", rc);
        }
    }
    return 0;
}
