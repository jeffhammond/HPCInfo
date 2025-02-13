module m

    !type, bind(C) :: ta
    type :: ta
        integer, allocatable :: m(:)
        integer :: i
    end type

    type, bind(C) :: tb
        integer  :: m(100)
        integer :: i
    end type

    !type, bind(C) :: ca
    type :: ca
        type(ta) :: a
        integer :: i
    end type

    type, bind(C) :: cb
        type(tb) :: b
        integer :: i
    end type

end module m

program main
    use m
    implicit none
    interface
        subroutine fa(xa) !bind(C)
            use m
            type(ca) :: xa
        end subroutine fa
    end interface
    interface
        subroutine fb(xb) bind(C)
            use m
            type(cb) :: xb
        end subroutine fb
    end interface
    type(ca) :: xa
    type(cb) :: xb
    integer :: n
    n = size(xb % b % m)
    allocate(xa % a % m(n))
    xa % i     = 1111111
    xa % a % i = 2222222
    xb % i     = 3333333
    xb % b % i = 4444444
    !print*,size(xa % a % m)
    print*,'loc(xa)        ',loc(xa)
    print*,'loc(xa % a)    ',loc(xa % a)
    print*,'loc(xa % i)    ',loc(xa % i)
    print*,'loc(xa % a % m)',loc(xa % a % m)
    print*,'loc(xa % a % i)',loc(xa % a % i)
    print*,'loc(xb)        ',loc(xb)
    print*,'loc(xb % b)    ',loc(xb % b)
    print*,'loc(xb % i)    ',loc(xb % i)
    print*,'loc(xb % b % m)',loc(xb % b % m)
    print*,'loc(xb % b % i)',loc(xb % b % i)
    call fa(xa)
    call fb(xb)
end program main
