module m

    !type, bind(C) :: t
    type :: t
        integer :: i
        double precision :: d
        integer :: j(10)
        real :: r(100)
        real, allocatable :: z(:)
    end type t

    interface
        subroutine foo(t) bind(C)
            implicit none
            type(*), dimension(..) :: t
        end subroutine foo
    end interface

end module m

program main
    use m
    implicit none
    type(t) :: x
    call foo(x)    
end program main
