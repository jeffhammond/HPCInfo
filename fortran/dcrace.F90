program main
  implicit none
  integer :: i, n
  real :: A(1000000)
  n = size(A)
  !do concurrent (i=1:n)
  !  A(i) = i
  !end do
  !print*,A(1),A(size(A))
  do concurrent (i=1:n) shared(A(1))
    A(1+mod(i,2)) = i
  end do
  print*,A(1:2)
end program main
