program main
  implicit none
  integer :: i, n, j, m
  real, allocatable :: x(:,:)
  n = 100
  m = 100
  allocate( x(m,n) )
  do i=1,n
    do concurrent (j=1:m)
      x(j,i) = j+i*n
    end do
  end do
  print*,x(i,j)
end program main
