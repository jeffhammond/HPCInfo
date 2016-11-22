      program test
      implicit none
      integer p1,p5,p6
      integer p1d,p5d,p6d
      integer i,j,jmax
      integer x,y,z
      p1d = 4
      p5d = 3
      p6d = 2
      i = 0
      do p1 = 1, p1d
        do p5 = 1, p5d
          do p6 = 1, p6d
            i = i + 1
            j = 1 + (p6-1) + (p5-1)*p6d + (p1-1)*p5d*p6d
            print*,'1:',i,j,p1,p5,p6
          enddo
        enddo
      enddo
      jmax = p1d*p5d*p6d
      i = 0
      do j = 1, jmax
        i = i + 1
        p6 = 1+mod(j-1,p6d)
        x  = 1+(j-p6)/p6d
        p5 = 1+mod(x-1,p5d)
        y  = 1+(x-p5)/p5d
        p1 = 1+mod(y-1,p1d)
        print*,'2:',i,j,p1,p5,p6
      enddo
      end program
