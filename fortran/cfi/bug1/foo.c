#include <stdio.h>
#include <stdlib.h>
#include "ISO_Fortran_binding.h"

char * get_type(CFI_type_t t)
{
    switch(t) {
        case CFI_type_cptr   : return "void *";                    break;
        case CFI_type_struct : return "interoperable C structure"; break;
        case CFI_type_other  : return "Not otherwise specified";   break;
        default              : abort();
    }
}

void foo(CFI_cdesc_t * d)
{
    printf("CFI_cdesc_t.type      = %s\n",  get_type(d->type));
}
