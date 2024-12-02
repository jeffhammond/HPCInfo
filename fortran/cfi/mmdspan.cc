#include <iostream>
#include <experimental/mdspan>
#include "ISO_Fortran_binding.h"

using namespace std::experimental;

template<typename T>
void mdspan_typed(CFI_cdesc_t * d)
{
    auto mds = mdspan(static_cast<T*>(d->base_addr));
}

extern "C" {

    void mdspan(CFI_cdesc_t * d) {
        switch (d->type)
        {
            case CFI_type_float:
                mdspan_typed<float>(d);
                break;
            default:
                printf("Unknown type\n");
                abort();
                break;
        }
    }

}
