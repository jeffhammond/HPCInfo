#include <stdio.h>
#include <stdlib.h>

extern int bar(int y1, int y2, int y3, int yy[1024])
{
    int z1 = 0;
    int z2 = 1;
    int z3 = 2;

    printf("bar:  f0=%p\n",__builtin_frame_address(0));

    printf("bar: &z3=%p\n",&z3);
    printf("bar: &z2=%p\n",&z2);
    printf("bar: &z1=%p\n",&z1);

    printf("bar: &y1=%p\n",&y1);
    printf("bar: &y2=%p\n",&y2);
    printf("bar: &y3=%p\n",&y3);

    printf("bar: &yy=%p\n",&yy);

    return y1+y2+y3;
}

extern int foo(int x)
{
    int z = 1;
    int zz[1024] = {0};
    printf("foo:  f0=%p\n",__builtin_frame_address(0));
    printf("foo: &zz=%p (end)\n",&zz[1023]);
    printf("foo: &zz=%p\n",&zz);
    printf("foo:  &z=%p\n",&z);
    printf("foo:  &x=%p\n",&x);
    printf("---------------\n");
    return bar(x,z,x+zz[1023],zz);
}

int main(int argc, char * argv[])
{
    int a = 17;
    printf("a=%d, &a=%p\n",a,&a);
    printf("================\n");
    foo(a);
    printf("================\n");
    return a;
}
