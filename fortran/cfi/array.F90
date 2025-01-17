module m

    interface
        subroutine foo(t) bind(C)
            implicit none
            type(*), dimension(..) :: t
        end subroutine foo
    end interface

    interface
        subroutine mdspan(t) bind(C)
            implicit none
            type(*), dimension(..) :: t
        end subroutine mdspan
    end interface

    contains

        subroutine bar(x)
            implicit none
            type(*), dimension(..) :: x
            call foo(x)
        end subroutine bar

        subroutine make(x)
            implicit none
            type(*), dimension(..) :: x
            call mdspan(x)
        end subroutine make

end module m

program main
    use m
    implicit none
    integer :: i
    real :: x(20)
    double precision :: y(10,5)

    x = [ (i, i = 1,size(x)) ]
    print*,'================'
    call bar(x)
    print*,x
    call mdspan(x)
    print*,'================'
    call bar(x(1:20:2))
    print*,x(1:20:2)
    call mdspan(x(1:20:2))
    print*,'================'

    y = reshape([ (i, i = 1,size(y,1)*size(y,2)) ],[size(y,1),size(y,2)])
    print*,'================'
    call bar(y)
    print*,y
    call mdspan(y)
    print*,'================'
    call bar(y(1:10:2,1:4))
    print*,y(1:10:2,1:4)
    call mdspan(y(1:10:2,1:4))

end program main
