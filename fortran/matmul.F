#define M 19
#define N 47
#define K 37

      program m
      implicit none
      real :: A(M,K)
      real :: B(K,N)
      real :: C(M,N)
      real :: D(M,N)
      integer :: x,y,z

      call random_number(A)
      call random_number(B)

      C = matmul(A,B)

      D = 0
      do concurrent(x=1:M, y=1:N)
        do z=1,K
          D(x,y) = D(x,y) + A(x,z) * B(z,y)
        enddo
      enddo

      print*,'max=',maxval(C-D)
      print*,'min=',minval(C-D)
    
      do x=1,M
        do y=1,N
          print*,x,y,C(x,y),D(x,y),C(x,y)-D(x,y),abs(C(x,y)-D(x,y)).ge.1e-5
        enddo
      enddo

      end program m
