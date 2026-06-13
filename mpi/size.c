#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int MPI_Type_size(int handle, int * size)
{
    (void)handle;
    *size = -32766;
    return 0;
}

void foo(size_t size)
{
    printf("foo: size=%zu\n", size);
}

int main(void)
{
    int size;
    MPI_Type_size(0,&size);
    printf("size=%d\n", size);
    foo( size * sizeof(double) );
    double *buf = malloc( size * sizeof(double) );
    memset(buf, 0xFF, size * sizeof(double) );
    printf("buf=%p, buf[0]=%f\n", (void*)buf, buf[0]);
    free(buf);
    return 0;
}
