module m

    interface
        pure subroutine print_error(code, message) bind(C,name="print_error")
            use iso_c_binding, only : c_int, c_char
            implicit none
            integer(c_int), intent(in), value :: code
            !character(kind=c_char), dimension(*), intent(in) :: message
            character(kind=c_char), dimension(..), intent(in) :: message
        end subroutine print_error
    end interface

    contains

    pure subroutine f(i)
        implicit none
        integer, intent(in) :: i
        if (i.eq.100) then
            call print_error(i,"bad things have happened")
        end if
    end subroutine f
end module m

program main
    use m
    implicit none
    integer :: i, n
    n = 1000
    do concurrent (i=1:n)
        call f(i)
    end do
end program main
