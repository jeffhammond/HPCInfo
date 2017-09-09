#include <stdio.h>
#include <ISO_Fortran_binding.h>

typedef struct {
    int i;
    int j;
    int k;
    double x;
    double y;
    double z;
} args;

void foo(args a)
{
    printf("i=%d\n",a.i);
    printf("j=%d\n",a.j);
    printf("k=%d\n",a.k);
    printf("x=%f\n",a.x);
    printf("y=%f\n",a.y);
    printf("z=%f\n",a.z);

}

void bar(CFI_cdesc_t * d)
{
    printf("CFI_cdesc_t.base_addr = %p\n",  d->base_addr);
    printf("CFI_cdesc_t.elem_len  = %zu\n", d->elem_len);
    printf("CFI_cdesc_t.version   = %d\n",  d->version);
    printf("CFI_cdesc_t.attribute = %td\n", d->attribute);
    printf("CFI_cdesc_t.rank      = %td\n", d->rank);
    printf("CFI_cdesc_t.type      = %td\n", d->type);
    printf("CFI_cdesc_t.dim[0].lb = %td\n", d->dim[0].lower_bound);
    printf("CFI_cdesc_t.dim[0].sm = %td\n", d->dim[0].sm);
    printf("CFI_cdesc_t.dim[0].xt = %td\n", d->dim[0].extent);


}
