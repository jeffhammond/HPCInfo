#include <iostream>
#include <experimental/mdspan>
#include "ISO_Fortran_binding.h"

using namespace std;
using namespace std::experimental;

template<typename T>
void mdspan_typed_contig(CFI_cdesc_t * d)
{
    std::cout << "mdspan_typed_config" << std::endl;
    auto mds = mdspan<T,
                      std::extents<CFI_index_t, dynamic_extent>,
                      layout_left>
                     (static_cast<T*>(d->base_addr),
                      dextents<CFI_index_t,1>{d->dim[0].extent});

    std::cout << "rank() = " << mds.rank() << "\n";
    std::cout << "size() = " << mds.size() << "\n";

    for (int i = 0; i < mds.rank(); i++)
        std::cout << "extent(" << i << ") = " << mds.extent(i) << "\n";

    for (size_t i = 0; i < mds.extent(0); i++)
        std::cout << i << "," << mds(i) << "\n";
}

template<typename T>
void mdspan_typed_1d(CFI_cdesc_t * d)
{
    std::cout << "mdspan_typed_1d" << std::endl;
    size_t extent = d->dim[0].extent;
    size_t stride = d->dim[0].sm / d->elem_len;

    std::dextents<size_t, 1> extents(extent);
    auto strides = std::array<size_t, 1>{{stride}};
    auto mapping = std::layout_stride::mapping(extents, strides);
    std::mdspan mds { static_cast<T*>(d->base_addr) , mapping };

    std::cout << "rank() = " << mds.rank() << "\n";
    std::cout << "size() = " << mds.size() << "\n";

    for (int i = 0; i < mds.rank(); i++)
        std::cout << "extent(" << i << ") = " << mds.extent(i) << "\n";

    for (size_t i = 0; i < mds.extent(0); i++)
        std::cout << i << "," << mds(i) << "\n";
}

template<typename T>
void mdspan_typed_2d(CFI_cdesc_t * d)
{
    std::cout << "mdspan_typed_2d" << std::endl;

    std::dextents<size_t, 2> extents(d->dim[0].extent,d->dim[1].extent);
    auto strides = std::array<size_t, 2>{{d->dim[0].sm / d->elem_len, d->dim[1].sm / d->elem_len}};
    auto mapping = std::layout_stride::mapping(extents, strides);
    std::mdspan mds { static_cast<T*>(d->base_addr) , mapping };

    std::cout << "rank() = " << mds.rank() << "\n";
    std::cout << "size() = " << mds.size() << "\n";

    for (int i = 0; i < mds.rank(); i++)
        std::cout << "extent(" << i << ") = " << mds.extent(i) << "\n";

    for (size_t j = 0; j < mds.extent(1); j++)
      for (size_t i = 0; i < mds.extent(0); i++)
        std::cout << i << "," << j << "," << mds(i,j) << "\n";
}

extern "C" {

    void mdspan(CFI_cdesc_t * d) {
        switch (d->type)
        {
            case CFI_type_float:
                switch (d->rank)
                {
                    case 1:
                        //mdspan_typed_contig<float>(d);
                        mdspan_typed_1d<float>(d);
                        break;
                    case 2:
                        mdspan_typed_2d<float>(d);
                        break;
                    default:
                        printf("Unknown rank\n");
                        abort();
                        break;
                }
                break;
            case CFI_type_double:
                switch (d->rank)
                {
                    case 1:
                        mdspan_typed_1d<double>(d);
                        break;
                    case 2:
                        mdspan_typed_2d<double>(d);
                        break;
                    default:
                        printf("Unknown rank\n");
                        abort();
                        break;
                }
                break;
            default:
                printf("Unknown type\n");
                abort();
                break;
        }
    }

}
