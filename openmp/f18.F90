program main
  implicit none
  integer, parameter :: n = 100
  real, dimension(n) :: x

  forall ( integer :: i = 1 : n )
    x(i) = i
  end forall

  do concurrent ( integer :: i = 1 : n )
    print*,x(i)
  end do

end program main
