module m
    contains
        subroutine user_logical_and(invec, inoutvec, len, datatype)
            use, intrinsic :: iso_c_binding, only : c_ptr
            use mpi_f08
            implicit none
            type(c_ptr), value :: invec, inoutvec
            integer :: len
            type(MPI_Datatype) :: datatype
            logical, dimension(:), pointer :: fpi, fpo
            character(len=MPI_MAX_OBJECT_NAME) :: name
            integer :: n
            if (datatype .ne. MPI_LOGICAL) then
                call MPI_Type_get_name(datatype, name, n)
                print*,'datatype (',name,') does not match arguments!'
                call MPI_Abort(MPI_COMM_WORLD,1)
            else
                call c_f_pointer(invec,fpi,[len])
                call c_f_pointer(inoutvec,fpo,[len])
            end if
            fpo = fpo .and. fpi
        end subroutine user_logical_and
end module m

program main
    use mpi_f08
    use m, only : user_logical_and
    implicit none
    integer :: i, me, np, n, argc, arglen, argerr
    integer, allocatable, dimension(:) :: a, b
    character(len=64) :: argtmp
    type(MPI_Op) :: op
    procedure(MPI_User_function), pointer :: fp => NULL()

    call MPI_Init()

    argc = command_argument_count()
    if (argc.ge.1) then
        call get_command_argument(1,argtmp,arglen,argerr)
        read(argtmp,'(i10)') n
    else
        n = 1000
    endif

    call MPI_Comm_rank(MPI_COMM_WORLD, me)
    call MPI_Comm_size(MPI_COMM_WORLD, np)

    fp => user_logical_and
    call MPI_Op_create(fp, .true., op)
    !call MPI_Op_create(user_logical_and, .true., op)

    allocate( a(n), b(n) )

    do i=1,n
        a(i) = merge(1,0,np .gt. 1)
    end do

    call MPI_Allreduce(a, b, n, MPI_LOGICAL, op, MPI_COMM_WORLD)

    do i=1,n
        if (b(i) .ne. merge(1,0,np .gt. 1)) then
            call MPI_Abort(MPI_COMM_SELF,np)
        end if
    end do

    deallocate( a, b )

    call MPI_Op_free(op)

    if (me.eq.0) print*,'OK'

    call MPI_Finalize()

end program main
