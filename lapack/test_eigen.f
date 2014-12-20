      program test
      implicit none
      integer n
      parameter (n=5)
      double precision a(n,n)
      double precision b(n,n)
      double precision eig(n)
      double precision wrk(3*n)
      integer info
      integer i,j,k
      double precision x
      call srand(999)
      do j=1,n
       x = rand(0)-0.5
       a(j,j) = x
       do i=1,j-1
        x = rand(0)-0.5
        a(i,j) = x
        a(j,i) = x
       enddo
      enddo   
      do j=1,n
       x = rand(0)
       b(j,j) = 1+x
       do i=1,j-1
        x = rand(0)/(i+j)
        b(i,j) = x
        b(j,i) = x
       enddo
      enddo   
      print*,'A='
      print*,a
      print*,'B='
      print*,b
c UPLO should not matter 
      call dsygv(1,'V','U',n,a,n,b,n,eig,wrk,3*n,info)
      print*,'A='
      print*,a
      print*,'B='
      print*,b
      print*,'lambda='
      print*,eig
      return
      end program test
