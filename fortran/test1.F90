program main
  integer :: k
  real :: x(10)
  do concurrent (k=1:2*size(x)) shared(x)
    x(mod(k,size(x))+1) = real(k)
    print*,k,mod(k,size(x))+1
  end do
end program main
