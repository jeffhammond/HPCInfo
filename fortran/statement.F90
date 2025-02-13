integer :: x(100)
do concurrent(integer :: i = 1 : 42 : 2) shared(x(1:100))
    print*,i
end do
end program
