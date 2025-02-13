module m
  contains
  elemental function iprod(x,y) result(z)
    use iso_fortran_env
    implicit none
    integer(kind=INT32), intent(in) :: x,y
    integer(kind=INT64) :: z
    z = int(x,INT64) * int(y,INT64)
  end function iprod
end module m

program test
  use m, only: iprod
  implicit none
  integer :: a,b
  a = 1000000000
  b = 1000000000
  print*,iprod(a,b)
end program test
