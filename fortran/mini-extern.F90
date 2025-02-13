module mpi
    use iso_c_binding
    integer(c_int), bind(C, name="MPIR_F08_MPI_IN_PLACE"), target :: MPI_IN_PLACE
    interface
        subroutine p() bind(C,name="p")
        end subroutine
    end interface
end module mpi

program main
    use mpi
    implicit none
    call p
    print*,'LOC(MPI_IN_PLACE)=',LOC(MPI_IN_PLACE)
end program main
