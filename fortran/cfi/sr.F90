subroutine bar2(x)
    implicit none
    type(*), dimension(..) :: x
    select rank(x)
        rank(0)
            stop 0
        rank default
            error stop 1
    end select
end subroutine bar2
