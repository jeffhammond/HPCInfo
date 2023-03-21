program main
    use mpi_f08
    implicit none
    integer :: me, np
    type(MPI_Comm), parameter :: world = MPI_COMM_WORLD
    call MPI_Init()
    call MPI_Comm_rank(world,me)
    call MPI_Comm_size(world,np)
    print*,"MPI F08: ", me, " of ", np
    call MPI_Finalize()
end program main
