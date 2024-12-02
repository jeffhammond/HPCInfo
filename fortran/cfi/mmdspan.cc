#include <iostream>
#include <experimental/mdspan>
#include "ISO_Fortran_binding.h"

using namespace std;
using namespace std::experimental;

template<typename T>
void mdspan_typed(CFI_cdesc_t * d)
{
    std::cout << "mdspan_typed" << std::endl;
    auto mds = mdspan<T,
                      std::extents<CFI_index_t, dynamic_extent>,
                      layout_left>
                     (static_cast<T*>(d->base_addr),
                      dextents<CFI_index_t,1>
                      {d->dim[0].extent});

    std::cout << "rank() = " << mds.rank() << "\n";
    std::cout << "size() = " << mds.size() << "\n";

    for (int i = 0; i < mds.rank(); i++)
        std::cout << "extent(" << i << ") = " << mds.extent(i) << "\n";

    for (size_t i = 0; i < mds.extent(0); i++)
        std::cout << i << "," << mds(i) << "\n";
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
