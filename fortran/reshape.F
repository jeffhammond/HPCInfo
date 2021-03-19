        program r
        implicit none
        integer :: i,j,k
        real ::  A(10,7)
        real ::  B(7,10)
        k = 1
        do i=1,10
         do j=1,7
           A(i,j) = k
           !B(j,i) = k
           k = k + 1
         enddo
        enddo
        B = 0
        B = reshape(A, (/7,10/), order=(/2,1/))
        do i=1,10
         do j=1,7
           print*,i,j,A(i,j),B(j,i)
         enddo
        enddo
        end program r
