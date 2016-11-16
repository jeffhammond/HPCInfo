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

#define DEBUG 1

static void handler(int sig, siginfo_t * info, void * state /* never use this */)
{
    /* sysconf probably should not be used in a signal handler.
     * the pagesize information should be encoded by the build system,
     * or we can be conservative and assume 4KiB. */
    size_t pagesize = sysconf(_SC_PAGESIZE);

#if DEBUG
    /* Using printf in a signal handler is evil but not
     * using it is a pain.  And printf works, so we will
     * use it because this is not production code anyways. */
    printf("handler: SIGSEGV at address %p\n", info->si_addr);
#endif

    /* this is the address the generated the fault */
    void * a = info->si_addr;

    /* these are at best unrelable in practice */
    void * l = info->si_lower;
    void * u = info->si_upper;
    
    /* b is the base address of the page that contains a. */
    /* there is surely a way to do this with bitwise logical ops instead... */
    void * b = (void*)(((intptr_t)a/(intptr_t)pagesize)*(intptr_t)pagesize);

    /* unprotect memory so the code that generated the fault can use it.
     * if nothing changes, fault will repeat forever. */
    int rc = mprotect(b, pagesize, PROT_READ | PROT_WRITE);

#if DEBUG
    if (rc) {
        printf("handler: mprotect failed - errno = %d\n", errno);
    }
    printf("handler: mprotect %p:%p PROT_READ | PROT_WRITE\n",b,b+pagesize);
    printf("handler: info addr=%p base=%p lower=%p, upper=%p\n",a,b,l,u);
    fflush(NULL);
#endif
    if (rc) {
        perror("mprotect failed; exit status is errno\n");
        exit(errno);
    }
}

int main(int argc, char* argv[])
{
    int rc = 0;

    size_t pagesize = sysconf(_SC_PAGESIZE);
    printf("pagesize = %zu\n", pagesize);

    /* setup SEGV handler */
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = handler;
    rc = sigaction(SIGSEGV, &sa, NULL);
    assert(rc==0);
    printf("sigaction set for SIGSEGV\n");

    /* it is good to know what these are. */
    printf("EACCES = %d\n",EACCES);
    printf("EINVAL = %d\n",EINVAL);
    printf("ENOMEM = %d\n",ENOMEM);

    /* allocate and initialize memory */
    const size_t n = pagesize;
    char * x = NULL;
    rc = posix_memalign((void**)&x,pagesize,n);
    if (x==NULL || rc!=0) {
        printf("posix_memalign ptr=%p, rc=%d, pagesize=%zu, bytes=%zu\n", x, rc, pagesize, n);
        abort();
    }
    memset(x,'f',n);
    printf("x=%p is allocated and bits are set to 'f'\n", x);

    /* set the page to be inaccessible so that any access will
     * generate a SEGV. */
    rc = mprotect(x,pagesize,PROT_NONE);
    assert(rc==0);
    printf("mprotect %p:%p PROT_NONE\n",x,x+pagesize);

    /* attempt to touch the protected page with an offset. */
    int offset = (argc>1) ? atoi(argv[1]) : 0;
    if (n-offset>0) memset(&(x[offset]),'a',n-offset);

    /* set the memory to read-only because that is all we need. */
    rc = mprotect(x,pagesize,PROT_READ);
    assert(rc==0);
    printf("mprotect %p:%p PROT_READ\n",x,x+pagesize);

    /* verify the results */
    printf("x = %s\n", x);

    free(x);

    printf("SUCCESS\n");
    return 0;
}
