#pragma once
#ifndef LARGE_COUNT_TYPE_HPP
#define LARGE_COUNT_TYPE_HPP

#include <cstdio>
#include <cstdlib>
#include <climits> // INT_MAX
#include <cstdint> // SIZE_MAX
#include <cassert>
#include <iostream>
#include <mpi.h>

inline void
SafeMpi( int mpiError )
{
    if( mpiError != MPI_SUCCESS )
    {
        char errorString[MPI_MAX_ERROR_STRING];
        int lengthOfErrorString;
        MPI_Error_string( mpiError, errorString, &lengthOfErrorString );
        std::cerr << std::string(errorString) << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
    }
}

class MpiLargelargeCountType
{
    public:
        MpiLargelargeCountType(MPI_Count largeCount, MPI_Datatype inType)
        {
            /* TODO C++-ify this as throw() */
            /* The largeCount has to fit into MPI_Aint for BigMPI to work. */
            assert(largeCount>=0);
            assert((unsigned long long)largeCount<(unsigned long long)SIZE_MAX);

            MPI_Count c = largeCount/INT_MAX;
            MPI_Count r = largeCount%INT_MAX;

            MPI_Datatype chunks;
            MPI_Type_vector(c, INT_MAX, INT_MAX, inType, &chunks);

            MPI_Datatype remainder;
            MPI_Type_contiguous(r, inType, &remainder);

            MPI_Aint lb /* unused */, extent;
            MPI_Type_get_extent(inType, &lb, &extent);

            MPI_Aint remdisp          = (MPI_Aint)c*INT_MAX*extent;
            int blocklengths[2]       = {1,1};
            MPI_Aint displacements[2] = {0,remdisp};
            MPI_Datatype types[2]     = {chunks,remainder};
            MPI_Type_create_struct(2, blocklengths, displacements, types, &(this->largeType));

            MPI_Type_free(&chunks);
            MPI_Type_free(&remainder);


        }

        ~MpiLargelargeCountType()
        {
            MPI_Type_free(&(this->largeType));
        }

    private:
        MPI_Datatype largeType;

};

#endif // LARGE_COUNT_TYPE_HPP
