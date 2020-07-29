#include <string>
#include <iostream>
#include <vector>

#include <mpi.h>

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int me=0, np=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&np);

    std::string in("x");
    for (int i=0; i<=me; ++i) {
        in += std::to_string(i);
    }

    std::cout << me << ": IN=" <<  in << " (" << in.size() << ")" << std::endl;

    std::string out(in);

    int tag = 0;
    if (me==0) {
        MPI_Status  status;
        for (int i=1; i<np; ++i) {
            int source = i;
            MPI_Probe(source, tag, MPI_COMM_WORLD, &status);
            int count = 0;
            MPI_Get_count(&status, MPI_CHAR, &count);
            std::string temp(count,' ');
            MPI_Recv(&(temp[0]), count, MPI_CHAR, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            out += temp;
        }
    }

    int dest = 0;
    MPI_Send(in.c_str(), (int)in.size(), MPI_CHAR, dest, tag, MPI_COMM_WORLD);

    if (me == 0) {
        std::cout << me << ": OUT=" <<  out << " (" << out.size() << ")" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
