/* code from https://software.intel.com/sites/default/files/Large_pages_mic_0.pdf */
#include <assert.h>
#include <stdlib.h>
#include <sys/mman.h>
#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define ALIGN_TO_PAGE_SIZE(x) \
    (((x) + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE)

void *malloc_huge_pages(size_t size)
{
    // Use 1 extra page to store allocation metadata
    // (libhugetlbfs is more efficient in this regard)
    size_t real_size = ALIGN_TO_PAGE_SIZE(size + HUGE_PAGE_SIZE);
    char *ptr = (char *)mmap(NULL, real_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS |
                  MAP_POPULATE | MAP_HUGETLB, -1, 0);
    if (ptr == MAP_FAILED) {
           // The mmap() call failed. Try to malloc instead
           ptr = (char *)malloc(real_size);
           if (ptr == NULL) return NULL;
           real_size = 0;
    }
    // Save real_size since mmunmap() requires a size parameter
    *((size_t *)ptr) = real_size;
    // Skip the page with metadata
    return ptr + HUGE_PAGE_SIZE;
}

void free_huge_pages(void *ptr)
{
    if (ptr == NULL) return;
    // Jump back to the page with metadata
    void *real_ptr = (char *)ptr - HUGE_PAGE_SIZE; // Read the original allocation size
    size_t real_size = *((size_t *)real_ptr);
    assert(real_size % HUGE_PAGE_SIZE == 0);
    if (real_size != 0)
           // The memory was allocated via mmap()
           // and must be deallocated via munmap()
           munmap(real_ptr, real_size);
    else
           // The memory was allocated via malloc()
           // and must be deallocated via free()
           free(real_ptr);
}
  
