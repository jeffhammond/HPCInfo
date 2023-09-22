program main
    implicit none
    integer :: i
    do concurrent (i=1:10)
        block
            real :: x
            x = i
            print*,x
        end block
    end do
end program main
