module m
    type :: magic_type
        integer :: junk
    end type

    type(magic_type) :: magic
    
    contains

    subroutine sub(arg,i)
        integer :: i
        class(*) :: arg
        select type (arg)
            type is (magic_type)
                call sub_magic(arg,i)
            class default
                call sub_normal(arg,i)
        end select
    end subroutine sub

    subroutine sub_normal(arg,i)
        integer :: i
        type(*), dimension(..) :: arg
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
