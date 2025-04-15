subroutine fx(n,a,c)
    implicit none
    integer :: n, i
    double precision :: a(n), b, c
    do i=1,n
        b = sin(a(n))
        c = max(c,cos(b))
    end do
end subroutine

subroutine fy(n,a,c)
    implicit none
    integer :: n, i
    double precision :: a(n), c
    do i=1,n
      block
        double precision :: b
        b = sin(a(n))
        c = max(c,cos(b))
      end block
    end do
end subroutine

subroutine fz(n,a,c)
    implicit none
    integer :: n, i
    double precision :: a(n), b, c
    block
      double precision :: b
      do i=1,n
          b = sin(a(n))
          c = max(c,cos(b))
      end do
    end block
end subroutine
