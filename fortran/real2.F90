program main
    use iso_fortran_env, only: INT64
    use iso_c_binding, only: c_loc, c_ptr
    implicit none
    real(4), target :: x
    real(4), pointer :: px
    type(c_ptr) :: cpx
    integer(8) :: vcpx
    x = 2
    print*,x,storage_size(x)
    print*,loc(x)
    px => x
    cpx = c_loc(px)
    vcpx = transfer(cpx,0_INT64)
    print*,vcpx
end program main
