real, dimension(0:100) :: x
real :: y(0:30)
real, allocatable, dimension(:) :: z
integer :: a(-3:3,-7:7)
allocate( z(0:1000) )
print*,size(x)
print*,size(y)
print*,size(z)
print*,size(A)
end program
