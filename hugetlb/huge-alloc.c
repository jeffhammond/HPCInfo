/* This is required to get MAP_HUGETLB, MAP_ANONYMOUS, MAP_POPULATE in C99 mode. */
/* It must be before all the heads, and not just before sys/mman.h. */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <sys/mman.h>
#ifndef NDEBUG
#include <errno.h>
#endif

/* we solve this with a hack for now */
#if 0
#if defined(LINUX_MAJOR) && defined(LINUX_MINOR) && \
    ((LINUX_MAJOR >= 4) || ((LINUX_MAJOR == 3) && (LINUX_MINOR >= 8)))
#define HUGE_PAGE_SIZE_OPTIONS_AVAILABLE
#endif
#endif

#ifndef NDEBUG
static inline int parse_error(void * rc)
{
    intptr_t irc = (intptr_t)rc;
    switch (irc) {
        case EACCES:
           fprintf(stderr,"A file descriptor refers to a non-regular file.  Or a file "
                          "mapping was requested, but fd is not open for reading.  Or "
                          "MAP_SHARED was requested and PROT_WRITE is set, but fd is not "
                          "open in read/write (O_RDWR) mode.  Or PROT_WRITE is set, but "
                          "the file is append-only.\n");
           return 1;
           break;
        case EAGAIN:
           fprintf(stderr,"The file has been locked, or too much memory has been locked"
                         "(see setrlimit(2)).\n");
           return 1;
           break;
        case EBADF:
           fprintf(stderr,"fd is not a valid file descriptor (and MAP_ANONYMOUS was not set\n");
           return 1;
           break;
        case EINVAL:
           fprintf(stderr,"We don't like addr, length, or offset (e.g., they are too "
                         "large, or not aligned on a page boundary).\n");
#if 0
           fprintf(stderr,"(since Linux 2.6.12) length was 0.\n");
           fprintf(stderr,"flags contained neither MAP_PRIVATE or MAP_SHARED, or contained both of these values.\n");
#endif
           return 1;
           break;
        case ENFILE:
           fprintf(stderr,"The system-wide limit on the total number of open files has been reached.\n");
           return 1;
           break;
        case ENODEV:
           fprintf(stderr,"The underlying filesystem of the specified file does not support memory mapping.\n");
           return 1;
           break;
        case ENOMEM:
           fprintf(stderr,"No memory is available.\n");
#if 0
           fprintf(stderr,"The process's maximum number of mappings would have been "
                          "exceeded.  This error can also occur for munmap(2), when "
                          "unmapping a region in the middle of an existing mapping, since "
                          "this results in two smaller mappings on either side of the "
                          "region being unmapped.\n");
#endif
           return 1;
           break;
        case EPERM:
           fprintf(stderr,"The prot argument asks for PROT_EXEC but the mapped area"
                          "belongs to a file on a filesystem that was mounted no-exec.\n");
#if 0
           fprintf(stderr,"The operation was prevented by a file seal; see fcntl(2).\n");
#endif
           return 1;
           break;
        case ETXTBSY:
           fprintf(stderr,"MAP_DENYWRITE was set but the object specified by fd is open for writing.\n");
           return 1;
           break;
        case EOVERFLOW:
           fprintf(stderr,"On 32-bit architecture together with the large file extension "
                          "(i.e., using 64-bit off_t): the number of pages used for "
                          "length plus number of pages used for offset would overflow "
                          "unsigned long (32 bits).\n");
           return 1;
           break;
        default:
           return 0;
           break;
    }
}
#endif

void * huge_alloc(size_t bytes, size_t pagesize)
{
    int rc = 0;
    char * ptr = NULL;

    switch (pagesize) {
        case (1UL<<30): /* 1G */
            {
                ptr = mmap(NULL, bytes,
                           PROT_READ | PROT_WRITE,     /* read-write memory   */
                           MAP_HUGETLB |
#ifdef MAP_HUGE_1GB
                           MAP_HUGE_1GB                /* 1G huge pages       */
#endif
                           MAP_PRIVATE |               /* not shared memory   */
                           MAP_ANONYMOUS |             /* no file backing     */ 
                           MAP_POPULATE,               /* pre-fault pages     */
                           -1, 0);                     /* ignored (anonymous) */
#ifndef NDEBUG
                rc = parse_error(ptr);
            }
#endif
            break;
        case (1UL<<21): /* 2M */
            {
                ptr = mmap(NULL, bytes,
                           PROT_READ | PROT_WRITE,     /* read-write memory   */
                           MAP_HUGETLB |
#ifdef MAP_HUGE_2MB
                           MAP_HUGE_2MB                /* 2M huge pages       */
#endif
                           MAP_PRIVATE |               /* not shared memory   */
                           MAP_ANONYMOUS |             /* no file backing     */ 
                           MAP_POPULATE,               /* pre-fault pages     */
                           -1, 0);                     /* ignored (anonymous) */
#ifndef NDEBUG
                rc = parse_error(ptr);
#endif
            }
            break;
        case (1UL<<12): /* 4K */
            {
                ptr = mmap(NULL, bytes,
                           PROT_READ | PROT_WRITE,     /* read-write memory   */
                           MAP_HUGETLB |
#ifdef MAP_HUGE_2MB
                           MAP_HUGE_2MB                /* 2M huge pages       */
#endif
                           MAP_PRIVATE |               /* not shared memory   */
                           MAP_ANONYMOUS |             /* no file backing     */ 
                           MAP_POPULATE,               /* pre-fault pages     */
                           -1, 0);                     /* ignored (anonymous) */
#ifndef NDEBUG
                rc = parse_error(ptr);
#endif
            }
            break;
        default:
            {
                fprintf(stderr, "huge_alloc: unsupported pagesize (%zu)\n", pagesize);
                /* if more than 2 MB is requested, use huge pages. */
                if (bytes >= (1UL<<21)) {
                    ptr = mmap(NULL, bytes,
                               PROT_READ | PROT_WRITE,     /* read-write memory   */
                               MAP_HUGETLB |
                               MAP_PRIVATE |               /* not shared memory   */
                               MAP_ANONYMOUS |             /* no file backing     */ 
                               MAP_POPULATE,               /* pre-fault pages     */
                               -1, 0);                     /* ignored (anonymous) */
                } else {
                    ptr = mmap(NULL, bytes,
                               PROT_READ | PROT_WRITE,     /* read-write memory   */
                               MAP_PRIVATE |               /* not shared memory   */
                               MAP_ANONYMOUS |             /* no file backing     */ 
                               MAP_POPULATE,               /* pre-fault pages     */
                               -1, 0);                     /* ignored (anonymous) */
                }
#ifndef NDEBUG
                rc = parse_error(ptr);
#endif
            }
            break;
    }
    if (rc) fprintf(stderr,"mmap(%zu) failed\n", bytes);
    return ptr;
}

void huge_free(void* ptr, size_t bytes)
{
    int rc = munmap(ptr, bytes);
    if (rc) fprintf(stderr,"munmap(%p,%zu) failed\n", ptr, bytes);
    return;
}

