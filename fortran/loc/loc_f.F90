module mycall
    interface
        pure function myloc(a) result(b) bind(C, name="myloc")
            use iso_c_binding
            type(*), intent(in) :: a
            integer(c_intptr_t) :: b
        end function myloc
    end interface
end module mycall

program test
    use iso_c_binding
    use mycall
    implicit none
    integer :: x
    integer :: y(2,2)
    print*,loc(x),myloc(x)
    print*,loc(y(1,1)),myloc(y(1,1))
    print*,loc(y(2,2)),myloc(y(2,2))
end program test


