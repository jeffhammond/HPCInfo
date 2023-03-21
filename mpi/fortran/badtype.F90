program main
    use mpi_f08
    implicit none
    call MPI_Init()
    call MPI_Barrier(MPI_REAL)
    call MPI_Finalize()
end program main
