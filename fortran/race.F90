program main
  integer :: k
  real :: x
  x = 0.0
  do concurrent (k=1:2) shared(x)
    x = real(k)
  end do
  print*,x
end program main

