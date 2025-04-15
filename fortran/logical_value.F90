module m

    interface
        subroutine set(i,v) bind(C,name="set")
            implicit none
            logical :: i
            integer :: v
        end subroutine set
    end interface

    contains

    subroutine foo(x,i)
        implicit none
        logical, intent(in) :: x
        integer, intent(in) :: i
        print*,'logical value',i
        if (x .eqv. .true.) then
            print*,'...is true'
        else if (x .eqv. .false.) then
            print*,'...is false'
        else
            print*,'...is neither true nor false'
        end if
        if (x .neqv. .true.) then
            print*,'...and not true'
        else if (x .neqv. .false.) then
            print*,'...and not false'
        else
            print*,'...and neither not true nor not false'
        end if
    end subroutine foo

end module m

program main
    use m
    implicit none
    logical :: x
    integer :: i
    i = 0
    call set(x,i)
    call foo(x,i)    
    i = 1
    call set(x,i)
    call foo(x,i)    
    i = -1
    call set(x,i)
    call foo(x,i)    
    i = 2
    call set(x,i)
    call foo(x,i)    
end program main
