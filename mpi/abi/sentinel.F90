module m
    integer :: magic
    contains
    subroutine sub(arg,i)
        integer :: i
        integer :: arg
        !type(*), dimension(..) :: arg
        if (loc(arg).eq.loc(magic)) print*,'LOC works ',i ! does not work with type(*)
    end subroutine sub
end module m

program main
    use m
    implicit none
    integer, parameter :: j = 1
    call sub(j,1)
    call sub(magic,2)
end program main
