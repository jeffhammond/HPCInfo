#include <mpi.h>

// PRIOR ART

int MPI_Grequest_start(MPI_Grequest_query_function *query_fn,
                       MPI_Grequest_free_function *free_fn,
                       MPI_Grequest_cancel_function *cancel_fn,
                       void *extra_state,
                       MPI_Request *request);

typedef int MPI_Grequest_free_function(void *extra_state);

int MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op);
int MPI_Op_create_c(MPI_User_function_c *user_fn, int commute, MPI_Op *op);

typedef void MPI_User_function(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);
typedef void MPI_User_function_c(void *invec, void *inoutvec, MPI_Count *len, MPI_Datatype *datatype);

// NEW FEATURe

// returns int unlike prior versions - the error code is returned from the calling reduction collective
// passes len and datatype by value since this is C-only - Fortran can use via bind(C) if necessary
// has extra state, which should not be updated by the function, to allow concurrent calls
typedef int MPI_User_reduce_function_c(void *invec, void *inoutvec,
                                       MPI_Count count, MPI_Datatype datatype, // pass by value because C only
                                       const void *extra_state);

// callback to free extra state once Op object is destroyed
typedef int MPI_User_free_function(void *extra_state);

// new op constructor
int MPI_Op_create_x(MPI_User_reduce_function_c *user_fn,
                    MPI_User_free_function *free_fn,
                    int commute,
                    void *extra_state,
                    MPI_Op *op);

// allows the user to access the state associated with an Op, in order to update it as appropriate
// fails if called with an op not created by MPI_Op_create_x.
int MPI_Op_get_state(MPI_Op op, void *extra_state); // argument is pointer to the void* state, like get_attr

/*

MPI_Op_create_x creates an op object associated with a new type of user-defined reduction,
MPI_User_reduce_function_c, which carries extra state.
Access to the extra state is read-only because the callback may be called concurrently.

The MPI_User_reduce_function_c callback passes count and datatype by value because this function
is designed for use in C, in order to support third-party language support and other use cases
where extra state is necessary.
It may be used from Fortran using the C-Fortran Interoperability (CFI) features of Fortran 2003.

MPI_User_free_function is called after MPI_Op_free has marked the object for deletion
and the implementation destroys the object, when the reference count reaches zero
(see Section 2.5.1 for details).

The user may update the state associated with an Op when it is not part of an active operation,
using the function MPI_Op_get_state.

*/
