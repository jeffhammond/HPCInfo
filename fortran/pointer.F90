program main
    implicit none
    integer, target :: a, b
    integer, pointer :: pa, ppa
    a = 1
    b = 2
    pa => a
    ppa => pa
    write(*,'(4a12)') 'a','b','pa','ppa'
    print*,a,b,pa,ppa
    pa => b
    print*,a,b,pa,ppa
end program main
