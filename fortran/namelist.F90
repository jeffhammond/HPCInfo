!CHARACTER (10) A, B (10)
!do concurrent(integer :: i = 1 : 42 : 2) !shared(A,B(1)) local(B(2:10))
!    B(2:10) = A(2:10)
!end do
integer, dimension(10), target :: A, B
integer, pointer :: x => B(1)
integer, dimension(:), pointer :: y => B(2:10)
A = 1
do concurrent(integer :: i = 1 : 42 : 2) shared(A) local(y)
   y = A(2:10)
end do
end program
