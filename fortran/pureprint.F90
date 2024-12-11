pure subroutine foo(x,y)
    implicit none
    real, intent(in) :: x
    real, intent(out) :: y
    if (x .lt. 0) then
        stop
    end if
    y = sqrt(x)
end subroutine foo
