subroutine foo(a,b)
    implicit none
    integer, value, intent(in) :: a
    integer, value :: b
    a = 21 ! can't assign
    b = 23
end subroutine foo
