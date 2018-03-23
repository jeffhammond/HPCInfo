#if __STDC_VERSION__ < 201112L

#error C11 is not supported.

#elif defined(__STDC_NO_THREADS__)

#error C11 is supported but threads are not.

#else

#include <stdio.h>
#include <threads.h>

int main(int argc, char **argv)
{
    thrd_t t;
    printf("success!\n");
    return 0;
}

#endif
