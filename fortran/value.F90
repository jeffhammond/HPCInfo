subroutine foo(x,y,z,a,b,c,d)
    implicit none
    integer, intent(in) :: x
    integer, intent(inout) :: y
    integer, intent(out) :: z
    integer, value, intent(in) :: a
    integer, intent(inout) :: b ! value not allowed beacuse intent(inout)
    integer, intent(out) :: c   ! value not allowed because intent(out)
    integer, value :: d         ! implicitly intent(in), because it can have no other intent
    integer :: i
    !integer, value :: j ! value not allowed because not dummy
    z = x + y
    c = a + b
    y = c
    b = z
    i = 17
    !j = 19
    !a = 21 ! can't assign
    d = 23
end subroutine foo
