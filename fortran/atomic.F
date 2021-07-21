program bomb
    implicit none
    real(8), allocatable, managed :: x(:)
    real(8) :: y1, y2
    integer :: n, i
    n = 1000*1000
    print*,'allocate'
    allocate( x(n) )
    print*,'initialize'
    do concurrent (i=1:n)
      x(i) = i
    end do
    print*,'OpenMP'
    y1 = 0
    do concurrent (i=1:n)
      !$omp atomic update
      y1 = y1 + x(i)
      !$omp end atomic
    end do
    print*,y1
    print*,'OpenACC'
    y2 = 0
    do concurrent (i=1:n)
      !$acc atomic update
      y2 = y2 + x(i)
      !$acc end atomic
    end do
    print*,y2
    print*,'Done'
end program bomb
