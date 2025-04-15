module m
    integer, parameter :: k = 2
end module m

program main
    use m
    implicit none
    integer(kind=k) :: i
end program main
