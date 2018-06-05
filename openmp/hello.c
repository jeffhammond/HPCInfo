#ifdef _OPENMP
# include <omp.h>
#else
# warning No OpenMP support!
#endif

int main(int argc, char * argv[])
{
    int nt = omp_get_max_threads();
    return nt;
}
