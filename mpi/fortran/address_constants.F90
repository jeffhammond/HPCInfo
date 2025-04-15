! /opt/homebrew/Cellar/open-mpi/5.0.3_1/bin/mpicc -c intercept.c && \
! /opt/homebrew/Cellar/open-mpi/5.0.3_1/bin/mpifort address_constants.F90 intercept.o && \
! /opt/homebrew/Cellar/open-mpi/5.0.3_1/bin/mpirun -n 1 ./a.out
module m
    interface
        subroutine intercept(s) bind(C,name="intercept")
            use mpi_f08
            implicit none
            type(MPI_Status) :: s
        end subroutine intercept
    end interface
end module m

program main
    use mpi_f08
    implicit none
    integer :: me, np
    type(MPI_Comm), parameter :: world = MPI_COMM_WORLD
    call MPI_Init()
    call intercept(MPI_STATUS_IGNORE)
    write(6,'(a25,i10)') 'LOC(MPI_STATUS_IGNORE) = ',LOC(MPI_STATUS_IGNORE)
    call MPI_Finalize()
end program main
