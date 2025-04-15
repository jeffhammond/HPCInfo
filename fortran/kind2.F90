program main
    use iso_c_binding
    use iso_fortran_env
    implicit none
    integer :: i
    integer*1 :: i1
    integer*2 :: i2
    integer*4 :: i4
    integer*8 :: i8
    integer(1) :: i_1
    integer(2) :: i_2
    integer(4) :: i_4
    integer(8) :: i_8
    integer(kind=c_int) :: ii
    integer(kind=c_int8_t) :: ii8
    integer(kind=c_int16_t) :: ii16
    integer(kind=c_int32_t) :: ii32
    integer(kind=c_int64_t) :: ii64
    print*,kind(i)
    print*,kind(i1)
    print*,kind(i2)
    print*,kind(i4)
    print*,kind(i8)
    print*,kind(i_1)
    print*,kind(i_2)
    print*,kind(i_4)
    print*,kind(i_8)
    print*,kind(ii)
    print*,kind(ii8)
    print*,kind(ii16)
    print*,kind(ii32)
    print*,kind(ii64)
end program main
