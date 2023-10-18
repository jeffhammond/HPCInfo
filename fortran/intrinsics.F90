program main
    implicit none

    ! arguments

#ifdef NONSTANDARD
    block
        integer :: n,i
        character(len=16) :: c
        n = iargc()
        print*,'iargc: ',n
        do i=0,n
            call getarg(i,c)
            print*,'getarg: ',i,c
        end do
    end block
#endif

    block
        integer :: n,i
        character(len=16) :: c
        n = command_argument_count()
        print*,'command_argument_count: ',n
        do i=0,n
            call get_command_argument(i,c)
            print*,'get_command_argument: ',i,c
        end do
        call get_command(c)
        print*,'get_command: ',c
    end block

    ! shell commands

#ifdef NONSTANDARD
    block
        integer :: s
        call system(command='true',status=s)
        print*,'true',s
        call system(command='false',status=s)
        print*,'false',s
    end block
#endif

    block
        integer :: s
        call execute_command_line(command='true',exitstat=s)
        print*,'true',s
        call execute_command_line(command='false',exitstat=s)
        print*,'false',s
    end block

    ! get current user

#ifdef NONSTANDARD
    block
        character(len=32) :: c = 'xxxxxxxxxxxxxx'
        call getlog(c)
        print*,'getlog: ',c
    end block
#endif

    block
        character(len=32) :: c = 'xxxxxxxxxxxxxx'
        call get_environment_variable('USER',c)
        print*,'USER: ',c
    end block

    ! date

#ifdef NONSTANDARD
    block
        character(len=32) :: c = 'xxxxxxxxxxxxxx'
        call fdate(c)
        print*,'date: ',c
    end block
#endif

    block
        character(len=8) :: d
        character(len=10) :: t
        character(len=5) :: z
        integer, dimension(8) :: v 
        call date_and_time(d,t,z,v)
        write(*,'(1x,a14,1x,a8,1x,a10,1x,a3,a5,1x,8i5)') 'date_and_time:',d,t,'UTC',z,v
    end block

    ! time

#ifdef NONSTANDARD
    block
        real(kind=4), dimension(2) :: v
        real(kind=4) :: t
        call etime(v,t)
        print*,'etime: ',v,t
    end block
#endif

    block
        real :: t
        call cpu_time(t)
        print*,'cpu_time: ',t
    end block    

    block
        integer(kind=4), dimension(3) :: c
        call system_clock(c(1),c(2),c(3))
        print*,'system_clock(4): ',c
    end block

    block
        integer(kind=8), dimension(3) :: c
        call system_clock(c(1),c(2),c(3))
        print*,'system_clock(8): ',c
    end block

end program main
