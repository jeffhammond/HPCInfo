module mpi
    use iso_c_binding
    !type(c_ptr), bind(C,name="MPI_F_IN_PLACE") :: MPI_IN_PLACE
    integer(c_int), bind(C, name="MPIR_F08_MPI_IN_PLACE"), target :: MPI_IN_PLACE
    interface
        subroutine p() bind(C,name="p")
        end subroutine
    end interface
    interface
        SUBROUTINE MPI_ALLREDUCE(SENDBUF, RECVBUF, COUNT, DATATYPE, OP, COMM, IERROR) &
                   bind(C,name="MPI_Allreduce")
            use iso_c_binding
            import :: MPI_IN_PLACE
            !DEC$ ATTRIBUTES NO_ARG_CHECK :: sendbuf,recvbuf
            !GCC$ ATTRIBUTES NO_ARG_CHECK :: sendbuf,recvbuf
            !$PRAGMA IGNORE_TKR sendbuf,recvbuf
            !DIR$ IGNORE_TKR sendbuf,recvbuf
            !IBM* IGNORE_TKR sendbuf,recvbuf
            INTEGER(kind=c_int) :: SENDBUF(*), RECVBUF(*)
            INTEGER(kind=c_int) :: COUNT, DATATYPE, OP, COMM, IERROR
        END SUBROUTINE MPI_ALLREDUCE
    end interface
end module mpi

program main
    use mpi
    implicit none
    real :: buf(100)
    integer :: ierror 
    call p
    buf = 17
    print*,'LOC(MPI_IN_PLACE)=',LOC(MPI_IN_PLACE)
    print*,'LOC(buf)=',LOC(buf)
    call MPI_ALLREDUCE(MPI_IN_PLACE,buf,1,2,3,4,ierror)
    print*,ierror
end program main
