#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

enum num { one=1, two=2, three=3 };
enum fat { four=4, five=5, big=INT_MAX };

int main(void)
{
    enum num x;
    enum fat y;
    printf("%zu\n",sizeof(x));
    printf("%zu\n",sizeof(y));
    return 0;
}
