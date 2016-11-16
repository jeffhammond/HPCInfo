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

size_t pagesize;

static void handler(int sig, siginfo_t * info, void * state)
{
    /* Using printf in a signal handler is evil but not
     * using it is a pain.  And printf works, so we will
     * use it because this is not production code anyways. */
    printf("handler: SIGSEGV at address %p\n", info->si_addr);
    void * a = info->si_addr;
    void * l = info->si_lower;
    void * u = info->si_upper;
    void * n = (void*)(((intptr_t)a/(intptr_t)pagesize)*(intptr_t)pagesize);
    int junk = mprotect(n,pagesize,PROT_WRITE);
    if (junk) printf("handler: mprotect failed - errno = %d\n", errno);
    printf("handler: mprotect %p:%p PROT_WRITE\n",n,n+pagesize);
    printf("handler: info addr=%p newaddr=%p lower=%p, upper=%p\n",a,n,l,u);
    fflush(NULL);
}

int main(int argc, char* argv[])
{
    int rc = 0;

    pagesize = sysconf(_SC_PAGESIZE);
    printf("pagesize = %zu\n", pagesize);

    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = handler;
    rc = sigaction(SIGSEGV, &sa, NULL);
    assert(rc==0);
    printf("sigaction set for SIGSEGV\n");

    printf("EACCES = %d\n",EACCES);
    printf("EINVAL = %d\n",EINVAL);
    printf("ENOMEM = %d\n",ENOMEM);

    const size_t n = pagesize;
    char * x = NULL;
    rc = posix_memalign((void**)&x,pagesize,n);
    if (x==NULL || rc!=0) {
        printf("posix_memalign ptr=%p, rc=%d, pagesize=%zu, bytes=%zu\n", x, rc, pagesize, n);
        abort();
    }
    memset(x,'f',n);
    printf("x=%p is allocated and bits are oned\n", x);

    rc = mprotect(x,pagesize,PROT_NONE);
    assert(rc==0);
    printf("mprotect %p:%p PROT_NONE\n",x,x+pagesize);

    int offset = (argc>1) ? atoi(argv[1]) : 0;
    if (n-offset>0) memset(&(x[offset]),'a',n-offset);

    rc = mprotect(x,pagesize,PROT_READ);
    assert(rc==0);
    printf("mprotect %p:%p PROT_READ\n",x,x+pagesize);

    printf("x = %s\n", x);

    free(x);
    printf("SUCCESS\n");
    return 0;
}
