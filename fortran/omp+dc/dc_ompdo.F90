program main
  implicit none
  integer :: i, n, j, m
  real, allocatable :: x(:,:)
  n = 100
  m = 100
  allocate( x(m,n) )
  do concurrent (j=1:m)
    !$omp parallel do
    do i=1,n
      x(i,j) = i+j*m
    end do
  end do
  print*,x(i,j)
end program main
