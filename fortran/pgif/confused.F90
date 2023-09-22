module confused
    use, intrinsic :: ISO_Fortran_env
    implicit none

    interface
        subroutine foo(buffer) bind(C,name="foo")
            use, intrinsic :: ISO_Fortran_env ! REQUIRED
            real(kind=REAL64), dimension(*) :: buffer
        end subroutine foo
    end interface

    contains

        subroutine bar(buffer) bind(C,name="bar")
            real(kind=REAL64), dimension(*) :: buffer
        end subroutine bar

end module confused

