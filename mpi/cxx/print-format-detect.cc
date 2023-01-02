#include <typeinfo>
#include <iostream>
#include <string>

#include <mpi.h>

int main(void)
{
    MPI_Count  c = 5;
    MPI_Aint   a = 6;
    MPI_Offset o = 7;

    std::string ff{"C=%"+std::string{typeid(MPI_Count).name()}+"\n"};
    printf(ff.c_str(),c);

    std::string gg{"A=%"+std::string{typeid(MPI_Aint).name()}+( std::is_signed<MPI_Aint>() ? "d" : "u")+"\n"};
    printf(gg.c_str(),a);

    std::string hh{"O=%"+std::string{typeid(MPI_Offset).name()}+"\n"};
    printf(hh.c_str(),o);

    return 0;
}
