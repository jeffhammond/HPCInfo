program main
    implicit none
    integer :: n
    real, allocatable, dimension(:) :: x
    n = 1000*1000*100
    allocate( x(n) )
    do concurrent (integer :: i = 1:n)
        x(i) = i
    end do
    deallocate( x )
end program main
