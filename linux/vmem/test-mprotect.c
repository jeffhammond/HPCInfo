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

char * t = "SIGSEGV at address ";
char * a = "                   ";
char * m = "                                         ";

ssize_t junk = 0;

size_t pagesize;

static void handler(int sig, siginfo_t * info, void * state)
{
#if 0
    sprintf(a,"%p",info->si_addr);
    strcat(m,t);
    strcat(m,a);
    junk = write(STDOUT_FILENO, m, strlen(m));
#else
    printf("handler: SIGSEGV at address %p\n", info->si_addr);
#endif
    void * x = info->si_addr;
    junk = mprotect(x,pagesize,PROT_WRITE);
    if (junk) abort();
    printf("handler: mprotect %p:%p PROT_WRITE\n",x,x+pagesize);
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

    size_t n = pagesize;
    char * x = NULL;
    rc = posix_memalign((void**)&x,pagesize,n);
    assert(x!=NULL && rc==0);
    memset(x,'f',n);
    printf("x=%p is allocated and bits are oned\n", x);

    rc = mprotect(x,pagesize,PROT_NONE);
    assert(rc==0);
    printf("mprotect %p:%p PROT_NONE\n",x,x+pagesize);

    memset(x,'a',n);

    rc = mprotect(x,pagesize,PROT_READ);
    assert(rc==0);
    printf("mprotect %p:%p PROT_READ\n",x,x+pagesize);

    printf("x = %s\n", x);

    free(x);
    printf("SUCCESS\n");
    return 0;
}
