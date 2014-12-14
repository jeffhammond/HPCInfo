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

class MpiTypeWrapper
{
    public:
        MpiTypeWrapper(MPI_Count largeCount, MPI_Datatype inType)
        {
            if ((unsigned long long)largeCount < (unsigned long long)INT_MAX) {
#ifdef DEBUGGING
                /* For debugging, I want to always create a datatype to 
                 * exercise the destructor properly. */
                /* Don't copy the mpiOn check here because I am not that dumb :-) */
                MPI_Type_contiguous(static_cast<int>(largeCount), inType, &(this->type));
                this->freeable = true;
                this->count    = static_cast<int>(largeCount);
                MPI_Type_commit(&(this->type));
#else
                this->freeable = false;
                this->count    = static_cast<int>(largeCount);
                this->type     = inType;
#endif
            } else {
                int mpiOn;
                MPI_Initialized(&mpiOn);
                if (!mpiOn) {
                    std::cerr << "Constructor called before MPI was initialized." << std::endl;
                }

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
                MPI_Type_create_struct(2, blocklengths, displacements, types, &(this->type));

                this->freeable = true;
                this->count    = static_cast<int>(largeCount);
                MPI_Type_commit(&(this->type));

                MPI_Type_free(&chunks);
                MPI_Type_free(&remainder);
            } // largeCount < INT_MAX
        } // ctor

        ~MpiTypeWrapper()
        {
            if (this->freeable) {
                int mpiOff;
                MPI_Finalized(&mpiOff);
                if (mpiOff) {
                    std::cerr << "Destructor called after MPI was finalized, "
                              << "presumably because you declared MpiTypeWrapper "
                              << "in the scope of main()." << std::endl;
                }
                MPI_Type_free(&(this->type));
            } // freeable
        }

        int GetMpiCount() { return this->count; }
        MPI_Datatype GetMpiDatatype() { return this->type; }

    private:
        bool freeable;
        int count;
        MPI_Datatype type;

};

#endif // LARGE_COUNT_TYPE_HPP
