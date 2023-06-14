function zdot(n,x,y) result(z) bind(C,name="zdot")
    implicit none
    integer, intent(in) :: n
    double complex, dimension(:), intent(in) :: x, y
    double complex :: z
    integer :: i
    z = (0.0d0,0.0d0)
    do i=1,n
        z = z + x(i) * y(i)
    end do
end function zdot
