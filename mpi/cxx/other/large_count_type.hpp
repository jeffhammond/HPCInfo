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

class MpiLargelargeCountType
{
    public:
        MpiLargelargeCountType(MPI_Count largeCount, MPI_Datatype inType)
        {
            int mpiOn, mpiOff;
            MPI_Initialized(&mpiOn);
            MPI_Finalized(&mpiOff);
            assert(mpiOn && !mpiOff);

            assert(largeCount>=0);

            if ((unsigned long long)largeCount < (unsigned long long)INT_MAX) {
                MPI_Type_contiguous(static_cast<int>(largeCount), inType, &(this->largeType));
                MPI_Type_commit(&(this->largeType));
            } else {
                /* TODO C++-ify this as throw() */
                /* The largeCount has to fit into MPI_Aint for BigMPI to work. */
                assert(static_cast<unsigned long long>(largeCount) < 
                       static_cast<unsigned long long>(SIZE_MAX));

                MPI_Count c = largeCount/INT_MAX;
                MPI_Count r = largeCount%INT_MAX;

                MPI_Datatype chunks;
                MPI_Type_vector(c, INT_MAX, INT_MAX, inType, &chunks);

                MPI_Datatype remainder;
                MPI_Type_contiguous(r, inType, &remainder);

                MPI_Aint lb /* unused */, extent;
                MPI_Type_get_extent(inType, &lb, &extent);

                MPI_Aint remdisp          = static_cast<MPI_Aint>(c)*static_cast<MPI_Aint>(INT_MAX)*extent;
                int blocklengths[2]       = {1,1};
                MPI_Aint displacements[2] = {0,remdisp};
                MPI_Datatype types[2]     = {chunks,remainder};
                MPI_Type_create_struct(2, blocklengths, displacements, types, &(this->largeType));
                MPI_Type_commit(&(this->largeType));

                MPI_Type_free(&chunks);
                MPI_Type_free(&remainder);
            } // largeCount < INT_MAX
        } // ctor

        ~MpiLargelargeCountType()
        {
            /* If user declares this in scope of main(), 
             * then the destructor will be called after MPI_Finalize() */
            int mpiOn, mpiOff;
            MPI_Initialized(&mpiOn);
            MPI_Finalized(&mpiOff);
            assert(mpiOn && !mpiOff);

            MPI_Type_free(&(this->largeType));
        }

        MPI_Datatype GetMpiDatatype()
        {
            return this->largeType;
        }

    private:
        MPI_Datatype largeType;

};

#endif // LARGE_COUNT_TYPE_HPP
