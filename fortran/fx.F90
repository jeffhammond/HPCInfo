subroutine fx(n,a,c)
    implicit none
    integer :: n, i
    double precision :: a(n), b, c
    do i=1,n
        b = sin(a(n))
        c = max(c,cos(b))
    end do
end subroutine
