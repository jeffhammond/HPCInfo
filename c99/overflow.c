#if !(defined(__GNUC__) && (__GNUC__ >= 5))
#warning GCC 5+ overflow build-ins probably not supported.
#endif

#include <stdlib.h>
#include <string.h>

void * calloc (size_t x, size_t y)
{
    size_t sz;
    if (__builtin_mul_overflow (x, y, &sz)) return NULL;
    void * ret = malloc(sz);
    if (ret) memset (ret, 0, sz);
    return ret;
}

