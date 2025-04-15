module m
  implicit real (a-z)
  !implicit none
end module m

program main
  !implicit none
  implicit integer (a-z)
  real :: x
  block
  use m
  a = 1.5
  print*,a
  end block
end program main
