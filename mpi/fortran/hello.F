program main
    implicit none
    include 'mpif.h'
    integer me, np, ierror
    integer world
    parameter (world = MPI_COMM_WORLD)
    call MPI_Init(ierror)
    call MPI_Comm_rank(world,me,ierror)
    call MPI_Comm_size(world,np,ierror)
    print*,"MPI F80: ", me, " of ", np
    call MPI_Finalize(ierror)
end program main
