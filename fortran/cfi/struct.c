#include <stdio.h>
#include <stddef.h>

struct t {
    double  d;
    int     i;
    int     j[10];
    float   r[100];
};

int main(void)
{
    printf("%zu\n",sizeof(struct t));
    printf("%zu\n",offsetof(struct t,d));
    printf("%zu\n",offsetof(struct t,i));
    printf("%zu\n",offsetof(struct t,j));
    printf("%zu\n",offsetof(struct t,r));
    return 0;
}
