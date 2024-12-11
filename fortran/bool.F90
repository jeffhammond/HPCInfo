module m
    interface
        subroutine p(i) bind(C,name="p")
            implicit none
            logical :: i
        end subroutine p
    end interface
end module m

program main
    use m
    implicit none
    print*,'.FALSE.'
    flush(6)
    call p(.FALSE.)
    print*,'.TRUE.'
    flush(6)
    call p(.TRUE.)
end program main
