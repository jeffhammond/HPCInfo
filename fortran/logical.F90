program bomb
    implicit none
    logical, allocatable :: x(:)
    logical :: y
    integer :: n, i
    n = 1000*1000
    print*,'allocate'
    allocate( x(n) )
    print*,'initialize'
    do concurrent (i=1:n)
      x(i) = mod(i,2).eq.0
    end do
    do concurrent (i=1:n)
       y = y .and. x(i)
    end do
    print*,y
    print*,'Done'
end program bomb
