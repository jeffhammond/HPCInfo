module m
    interface
        subroutine set(i,v) bind(C,name="set")
            implicit none
            logical :: i
            integer :: v
        end subroutine set
    end interface
end module m

program main
    use m
    implicit none
end program main
