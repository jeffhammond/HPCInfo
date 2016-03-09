#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
int main(void)
{
    size_t align = 128;
    size_t bytes = 100*align;
    char * buffer = aligned_alloc(align,bytes);
    printf("buffer = %p\n",buffer);
    memset(buffer,255,bytes);
    free(buffer);
    return 0;
}
#else
#error You need C11 compiler.
#endif


