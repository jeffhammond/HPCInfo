program main
  use iso_fortran_env
  use omp_lib
  implicit none

  integer, parameter :: iterations = 1000
  integer, parameter :: elements = 1024

  integer(kind=INT32) :: i32(elements)
  integer(kind=INT64) :: i64(elements)
  real(kind=REAL32)   :: r32(elements)
  real(kind=REAL64)   :: r64(elements)

  integer(kind=INT32) :: i32_min
  integer(kind=INT64) :: i64_min
  real(kind=REAL32)   :: r32_min
  real(kind=REAL64)   :: r64_min

  integer(kind=INT32) :: i32_max
  integer(kind=INT64) :: i64_max
  real(kind=REAL32)   :: r32_max
  real(kind=REAL64)   :: r64_max

  !$omp parallel

  !$omp end parallel

end program main
