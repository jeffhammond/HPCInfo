module m
  !implicit real (a-z)
  implicit none
end module m

program main
  use m
  !implicit none
  implicit real (a-z)
  a = 1.0
end program main
