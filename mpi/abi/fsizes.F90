program main
    use iso_fortran_env
    implicit none
    logical :: l
    integer :: i
    real :: r
    double precision :: d
    complex :: c
    print*,'NUMERIC_STORAGE_SIZE=',NUMERIC_STORAGE_SIZE
    print*,'STORAGE_SIZE(LOGICAL)=',STORAGE_SIZE(l)
    print*,'STORAGE_SIZE(INTEGER)=',STORAGE_SIZE(i)
    print*,'STORAGE_SIZE(REAL)=',STORAGE_SIZE(r)
    print*,'STORAGE_SIZE(DOUBLE PRECISION)=',STORAGE_SIZE(d)
    print*,'STORAGE_SIZE(COMPLEX)=',STORAGE_SIZE(c)
end program main 
