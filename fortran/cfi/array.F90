module m

    !type :: t
    type, bind(C) :: t          ! 456
        double precision :: d   !   8
        integer :: i            !   4
        integer :: j(10)        !  40
        real :: r(100)          ! 400
                                !   4 padding
        !real, allocatable :: z(:)
    end type t

    interface
        subroutine foo(t) bind(C)
            implicit none
            type(*), dimension(..) :: t
        end subroutine foo
    end interface

    contains

        subroutine bar(x)
            implicit none
            type(*), dimension(..) :: x
            call foo(x)
        end subroutine bar

end module m

program main
    use m
    implicit none
    type(t) :: x
    !real :: x(20)
    call bar(x)
end program main
