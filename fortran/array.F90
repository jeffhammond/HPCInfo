      program test
      real x(-1:1)
!      x(-1) = -1
!      x(0)  = 0
!      x(1)  = 1
      print*,"lower bound = ", LBOUND(x)
      print*,"upper bound = ", UBOUND(x)
      print*,"size        = ", SIZE(x)
      end program test
