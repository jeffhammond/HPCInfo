#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>

#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

int main(int argc, char* argv[])
{
    int n = (argc>1) ? atoi(argv[1]) : 30;
    size_t bytes = 1UL<<n;

    //int a = 137777;
    //printf("a=%d\n", a);
    //int * pa = &a;
    int * pa = malloc(4096);
    printf("pa=%p\n", pa);
    intptr_t ipa = (intptr_t)pa;
    printf("ipa=%zu\n", ipa);
    intptr_t xipa = ipa | (intptr_t)0xFFFF000000000000ULL;
    printf("xipa=%zu\n", xipa);
    int * xpa = (int *)xipa;
    printf("xpa=%p\n", xpa);
    {
        int prot  = PROT_READ | PROT_WRITE;      /* read-write memory    */
        int flags = MAP_ANONYMOUS |              /* no file backing      */
                    MAP_NORESERVE |              /* no swap space        */
                    MAP_PRIVATE;                 /* not shared memory    */
                    //MAP_FIXED;                   /* at this address      */
        char * ptr = mmap(NULL, bytes, prot, flags, -1, 0);
        printf("ptr=%p\n", ptr);
        printf("(int)ptr=%zu\n", (intptr_t)ptr);
        memset(ptr,0,1UL<<12);
        int rc = munmap(ptr,bytes);
        printf("munmap=%d\n", rc);
    }
    int * zpa = (int *) ( (intptr_t)xpa & (intptr_t)0x0000FFFFFFFFFFFFULL );
    printf("zpa=%p\n", zpa);
    //int za = *zpa;
    //printf("z=%d\n", za);
    //printf("%p %zu %zu %p %d\n", pa, ipa, xipa, xpa, xa);
    return 0;
}
