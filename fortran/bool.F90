module m
    interface
        subroutine p(i) bind(C,name="p")
            implicit none
            logical :: i
        end subroutine p
    end interface
!    interface
!        subroutine q(i) bind(C,name="q")
!            implicit none
!            type(*), dimension(..) :: i
!        end subroutine q
!    end interface
end module m

program main
    use m
    implicit none
    print*,'.FALSE.'
    flush(6)
    call p(.FALSE.)
!    call q(.FALSE.)
    print*,'.TRUE.'
    flush(6)
    call p(.TRUE.)
!    call q(.TRUE.)
end program main
