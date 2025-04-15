program main
    implicit none
    integer, target :: a, b
    integer, pointer :: pa, pb
    write(*,'(4a12)') 'a','b','pa','pb'
    a = 1
    b = 2
    pa => a
    pb => b
    print*,a,b,pa,pb
    !pb => pa
    !print*,a,b,pa,pb
    pb = pa
    print*,a,b,pa,pb
end program main
