#include <iostream>
#include <experimental/mdspan>
#include "ISO_Fortran_binding.h"

using namespace std::experimental;

template<typename T>
void mdspan_typed(CFI_cdesc_t * d)
{
    std::cout << "mdspan_typed" << std::endl;
    auto mds = mdspan(static_cast<T*>(d->base_addr), layout_left);
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
