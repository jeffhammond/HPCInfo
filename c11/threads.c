#if __STDC_VERSION__ < 201112L

#error C11 is not supported.

#elif defined(__STDC_NO_THREADS__)

#error C11 is supported but threads are not.

#else

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <threads.h>

#include <assert.h>

int main(int argc, char **argv)
{
    printf("success!\n");

    return 0;
}

#endif
