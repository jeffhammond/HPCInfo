#include <stdio.h>
#include <stdlib.h>
#include "ISO_Fortran_binding.h"

char * get_attr(CFI_attribute_t a);
char * get_type(CFI_type_t t);

void p(int * i)
{
    printf("sint: %d\n",*i);
    printf("uint: %u\n",(unsigned)*i);
    printf("hex:  %x\n",*i);
    printf("addr:  %p\n",i);
}

void q(CFI_cdesc_t * d)
{
    printf("CFI_cdesc_t           = %p\n",  d);
    printf("CFI_cdesc_t.base_addr = %p\n",  d->base_addr);
    printf("CFI_cdesc_t.elem_len  = %zu bytes\n", d->elem_len);
    printf("CFI_cdesc_t.version   = %d\n",  d->version);
    printf("CFI_cdesc_t.attribute = %s\n",  get_attr(d->attribute));
    printf("CFI_cdesc_t.rank      = %d\n",  (int)d->rank);
    printf("CFI_cdesc_t.type      = %s\n",  get_type(d->type));
    if (d->rank > 0) {
        printf("CFI_cdesc_t.dim[0].lb = %td\n", (ptrdiff_t)d->dim[0].lower_bound);
        printf("CFI_cdesc_t.dim[0].sm = %td\n", (ptrdiff_t)d->dim[0].sm);
        printf("CFI_cdesc_t.dim[0].xt = %td\n", (ptrdiff_t)d->dim[0].extent);
    }
}

char * get_attr(CFI_attribute_t a)
{
    switch(a) {
        case CFI_attribute_pointer:     return "data pointer";      break;
        case CFI_attribute_allocatable: return "allocatable";       break;
        case CFI_attribute_other:       return "other";             break;
        default:                        return "unknown attribute"; break;
    }
}

char * get_type(CFI_type_t t)
{
    switch(t) {
#if 0
        case CFI_type_Integer              :    return "Integer";                     break;
        case CFI_type_Real                 :    return "Real";                        break;
        case CFI_type_Complex              :    return "Complex";                     break;
        case CFI_type_Logical              :    return "Logical";                     break;
#endif
        //case CFI_type_signed_char          :     return "signed char";                break;
        //case CFI_type_short                :     return "short int";                  break;
        //case CFI_type_int                  :     return "int";                        break;
        //case CFI_type_long                 :     return "long int";                   break;
        //case CFI_type_long_long            :     return "long long int";              break;
        //case CFI_type_size_t               :     return "size_t";                     break;
        case CFI_type_int8_t               :     return "int8_t";                     break;
        case CFI_type_int16_t              :     return "int16_t";                    break;
        case CFI_type_int32_t              :     return "int32_t";                    break;
        case CFI_type_int64_t              :     return "int64_t";                    break;
        //case CFI_type_int_least8_t         :     return "int_least8_t";               break;
        //case CFI_type_int_least16_t        :     return "int_least16_t";              break;
        //case CFI_type_int_least32_t        :     return "int_least32_t";              break;
        //case CFI_type_int_least64_t        :     return "int_least64_t";              break;
        //case CFI_type_int_fast8_t          :     return "int_fast8_t";                break;
        //case CFI_type_int_fast16_t         :     return "int_fast16_t";               break;
        //case CFI_type_int_fast32_t         :     return "int_fast32_t";               break;
        //case CFI_type_int_fast64_t         :     return "int_fast64_t";               break;
        //case CFI_type_intmax_t             :     return "intmax_t";                   break;
        //case CFI_type_intptr_t             :     return "intptr_t";                   break;
        //case CFI_type_ptrdiff_t            :     return "ptrdiff_t";                  break;
        case CFI_type_float                :     return "float";                      break;
        case CFI_type_double               :     return "double";                     break;
        //case CFI_type_long_double          :     return "long double";                break;
        case CFI_type_float_Complex        :     return "float _Complex";             break;
        case CFI_type_double_Complex       :     return "double _Complex";            break;
        //case CFI_type_long_double_Complex  :     return "long double _Complex";       break;
        case CFI_type_Bool                 :     return "_Bool";                      break;
        case CFI_type_char                 :     return "char";                       break;
        case CFI_type_cptr                 :     return "void *";                     break;
        case CFI_type_struct               :     return "interoperable C structure";  break;
        case CFI_type_other                :     return "Not otherwise specified";    break;
        default: {
#ifdef GFORTRAN
            // gfortran-specific
            int i = t & CFI_type_mask;
            int k = (t-i)  >> CFI_type_kind_shift;
            printf("   unknown type is %d\n", t);
            //printf("CFI_type_kind_shift = %d\n", CFI_type_kind_shift);
            //printf("CFI_type_mask = %d\n", CFI_type_mask);
            printf("   CFI_type_Integer    = %d\n", CFI_type_Integer);
            printf("   CFI_type_Logical    = %d\n", CFI_type_Logical);
            printf("   CFI_type_Real       = %d\n", CFI_type_Real   );
            printf("   CFI_type_Complex    = %d\n", CFI_type_Complex);
            printf("   CFI_intrinsic_type  = %d (see above)\n", i);
            printf("   CFI_type_kind       = %d (storage size)\n", k);
#endif
            return "unknown type";
            break;
        }
    }
}
