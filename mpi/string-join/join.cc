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

    size_t mysize = in.size();
    size_t totalsize = 0;

    assert(sizeof(size_t)==sizeof(int64_t));
    MPI_Reduce(&mysize, &totalsize, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    std::string out;
    out.resize(totalsize);

    int mysizeint = (int)mysize;
    std::vector<int> counts(np,-1);
    MPI_Gather(&mysizeint, 1, MPI_INT, &(counts[0]), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(np,0);
    for (int j=0; j<np; ++j) {
        for (int i=0; i<j; ++i) {
            displs[j] += counts[i];
        }
    }

    MPI_Gatherv(in.c_str(), mysizeint, MPI_CHAR, &(out[0]), &(counts[0]), &(displs[0]), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (me == 0) {
        std::cout << me << ": OUT=" <<  out << " (" << out.size() << ")" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
