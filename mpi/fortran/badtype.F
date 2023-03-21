program main
    implicit none
    include 'mpif.h'
    integer ierror
    call MPI_Init(ierror)
    call MPI_Barrier(MPI_REAL,ierror)
    call MPI_Finalize(ierror)
end program main
