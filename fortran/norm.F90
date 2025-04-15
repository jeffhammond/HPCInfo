program norm
    implicit none
    double precision, allocatable :: x(:)
    double precision :: y, z
    integer, parameter :: n = 100000
    allocate( x(n) )
    call random_number(x)
    y = dsqrt(dot_product(x,x))
    z = norm2(x)
    print*,y,z
end program norm
