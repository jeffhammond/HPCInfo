#include <typeinfo>
#include <iostream>

#include <mpi.h>

int main(void)
{
    MPI_Count  c = 0;
    MPI_Aint   a = 0;
    MPI_Offset o = 0;
    std::cout << "MPI_Count  is " << typeid(MPI_Count).name() << std::endl;
    std::cout << "MPI_Aint   is " << typeid(MPI_Aint).name() << std::endl;
    std::cout << "MPI_Offset is " << typeid(MPI_Offset).name() << std::endl;
    std::cout << "MPI_Count  is " << typeid(c).name() << std::endl;
    std::cout << "MPI_Aint   is " << typeid(a).name() << std::endl;
    std::cout << "MPI_Offset is " << typeid(o).name() << std::endl;
    return 0;
}
