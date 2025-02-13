module cfi
    implicit none
    interface
        subroutine bar(buffer) bind(C,name="bar")
            integer(4), dimension(:,:,:) :: buffer
        end subroutine bar
    end interface
end module cfi 

program test
    use cfi
    implicit none
    block
        integer(4), dimension( 3:7, 2:8, 1:9) :: q5
        q5 = -1
        call bar(q5(3:7:2,2:8:2,1:9:2))
    end block

end program test
