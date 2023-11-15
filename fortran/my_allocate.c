#include <stdlib.h>

void my_allocate(size_t size, size_t align, void ** baseptr)
{
    *baseptr = aligned_alloc(align, size);
}
