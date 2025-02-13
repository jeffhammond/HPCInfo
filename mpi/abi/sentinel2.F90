module m
    type :: magic_type
        integer :: junk
    end type

    type(magic_type) :: magic
    
    interface sub
        !procedure sub_normal, sub_magic
        module procedure sub_normal, sub_magic
        !module procedure sub_magic
    end interface sub

    contains

    subroutine sub_normal(arg,i)
        integer :: i
        !type(*), dimension(..) :: arg ! doesn't work
        class(*), dimension(..) :: arg
        print*,'NORMAL'
    end subroutine sub_normal

    subroutine sub_magic(arg,i)
        integer :: i
        type(magic_type) :: arg
        print*,'MAGIC worked'
    end subroutine sub_magic

end module m

program main
    use m
    implicit none
    integer, parameter :: j = 1
    call sub(j,1)
    call sub(magic,2)
end program main
