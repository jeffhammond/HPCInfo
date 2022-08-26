module pgif
    use, intrinsic :: ISO_C_binding
    use, intrinsic :: ISO_Fortran_env

    contains

        subroutine foo(buffer,m,n) bind(C,name="foo")
            implicit none
            real(kind=REAL64), dimension(*) :: buffer
            integer :: m, n
        end subroutine foo

        subroutine bar(buffer,m,n) bind(C,name="bar")
            implicit none
            real(kind=REAL64), dimension(..) :: buffer
            integer :: m, n
        end subroutine bar

end module pgif

