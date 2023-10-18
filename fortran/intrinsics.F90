program main
    implicit none

    ! shell commands

    block
        integer :: s
        call system(command='true',status=s)
        print*,'true',s
        call system(command='false',status=s)
        print*,'false',s
    end block

    block
        integer :: s
        call execute_command_line(command='true',exitstat=s)
        print*,'true',s
        call execute_command_line(command='false',exitstat=s)
        print*,'false',s
    end block

    ! get current user

    block
        character(len=32) :: c = 'xxxxxxxxxxxxxx'
        call getlog(c)
        print*,'getlog: ',c
    end block

    block
        character(len=32) :: c = 'xxxxxxxxxxxxxx'
        call get_environment_variable('USER',c)
        print*,'USER: ',c
    end block

    ! date

    block
        character(len=32) :: c = 'xxxxxxxxxxxxxx'
        call fdate(c)
        print*,'date: ',c
    end block

    block
        character(len=8) :: d
        character(len=10) :: t
        character(len=5) :: z
        integer, dimension(8) :: v 
        call date_and_time(d,t,z,v)
        write(*,'(1x,a14,1x,a8,1x,a10,1x,a3,a5,1x,8i5)') 'date_and_time:',d,t,'UTC',z,v
    end block


end program main
