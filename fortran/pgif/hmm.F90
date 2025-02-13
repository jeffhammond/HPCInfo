module m

    type, bind(C) :: ta
        integer, allocatable :: m(:)
        integer :: i
    end type

    type, bind(C) :: tb
        integer  :: m(100)
        integer :: i
    end type

    type, bind(C) :: ca
        type(ta) :: a
        integer :: i
    end type

    type, bind(C) :: cb
        type(tb) :: b
        integer :: i
    end type

    interface
        subroutine fa(a) bind(C)
            type(ta) :: a
        end subroutine fa
    end interface

    interface
        subroutine fb(b) bind(C)
            type(cb) :: b
        end subroutine fb
    end interface

end module m

program main
    use m
    implicit none
end program main
