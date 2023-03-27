/*
The function takes three arguments: nnodes (total number of nodes), ndims (number of dimensions), and dims (array of dimensions). The function returns MPI_SUCCESS if it succeeds or an error code if it fails.

The function first checks if ndims is zero and returns MPI_ERR_DIMS if it is. Then it calculates the product of all the non-zero elements of dims to get the total number of nodes in the grid.

Next, it calculates the remaining nodes that need to be distributed among the dimensions that have a zero value in dims. It does this by dividing nnodes by the product of the non-zero elements of dims and then decrementing remain until it finds a value that evenly divides the remaining nodes.

Then it sets the dimensions that have a zero value in dims to the calculated remain value and updates the product of the dimensions accordingly.

Finally, it sorts the dimensions in non-increasing order and returns MPI_SUCCESS.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int XXX_Dims_create(int nnodes, int ndims, int dims[]) {
    int i, factor, remain;
    
    if (ndims == 0) {
        return MPI_ERR_DIMS;
    }
    
    factor = 1;
    for (i = 0; i < ndims; i++) {
        if (dims[i] < 0) {
            return MPI_ERR_ARG;
        } else if (dims[i] > 0) {
            factor *= dims[i];
        }
    }
    
    remain = nnodes / factor;
    for (i = 0; i < ndims; i++) {
        if (dims[i] == 0) {
            while (remain > 1 && (nnodes % (factor * remain)) != 0) {
                remain--;
            }
            dims[i] = remain;
            factor *= remain;
            remain = nnodes / factor;
        }
    }
    
    for (i = 0; i < ndims; i++) {
        int j, max_index = i;
        int max_val = dims[i];
        for (j = i + 1; j < ndims; j++) {
            if (dims[j] > max_val) {
                max_index = j;
                max_val = dims[j];
            }
        }
        if (max_index != i) {
            int tmp = dims[i];
            dims[i] = dims[max_index];
            dims[max_index] = tmp;
        }
    }
    
    return MPI_SUCCESS;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    if (argc < 4) MPI_Abort(MPI_COMM_WORLD, argc);
    {
        int nnodes = atoi(argv[1]);
        printf("IN nnodes = %d\n", nnodes);
        int ndims  = atoi(argv[2]);
        printf("IN ndims = %d\n", ndims);
        int * dims = calloc(ndims,sizeof(int));
        for (int i=0; i<ndims; i++) {
            dims[i] = atoi(argv[3+i]);
            printf("IN dims[%d] = %d\n", i, dims[i]);
        }

        MPI_Dims_create(nnodes,ndims,dims);
        for (int i=0; i<ndims; i++) {
            printf("MPI dims[%d] = %d\n", i, dims[i]);
        }

        for (int i=0; i<ndims; i++) {
            dims[i] = atoi(argv[3+i]);
            printf("IN dims[%d] = %d\n", i, dims[i]);
        }

        XXX_Dims_create(nnodes,ndims,dims);
        for (int i=0; i<ndims; i++) {
            printf("XXX dims[%d] = %d\n", i, dims[i]);
        }
        free(dims);
    }
    MPI_Finalize();
    return 0;
}
