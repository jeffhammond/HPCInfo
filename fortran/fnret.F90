integer function foo(i)
    implicit none
    integer, intent(in) :: i
    print*,'i=',i
    foo = i*100
    return
end function foo

program main
    implicit none
    integer :: r
    integer :: foo
    r = foo(0)
    print*,'r=',r
    r = foo(1)
    print*,'r=',r
end program main
