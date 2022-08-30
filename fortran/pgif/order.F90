program t
  interface
    subroutine foo(A,B)
      real, dimension(:,:,:) :: A
      !pgi$ ignore_tkr(c) A
      real, dimension(:,:,:) :: B
      !pgi$ ignore_tkr(c) B
    end subroutine
  end interface
  real :: A(10,11,12)
  real :: B(10,11,12)
  call foo(A(2:10:2,1:11:2,2:12:2),B(1:10:1,1:11:1,1:12:1))
end program

subroutine foo(A, B, D1, D2)
  real, dimension (*) :: A, B
  integer(8), dimension(*) :: D1, D2
  print *,D1(1)
  print *,"rank",D1(2)
  print *,"offset",D1(8)
  print *,"lbound 1",D1(11)
  print *,"extent 1",D1(12)
  print *,"stride 1",D1(15)
  print *,"lbound 2",D1(11+6)
  print *,"extent 2",D1(12+6)
  print *,"stride 2",D1(15+6)
  print *,"lbound 3",D1(11+12)
  print *,"extent 3",D1(12+12)
  print *,"stride 3",D1(15+12)
  print *,D2(1)
  print *,"rank",D2(2)
  print *,"offset",D2(8)
  print *,"lbound 1",D2(11)
  print *,"extent 1",D2(12)
  print *,"stride 1",D2(15)
  print *,"lbound 2",D2(11+6)
  print *,"extent 2",D2(12+6)
  print *,"stride 2",D2(15+6)
  print *,"lbound 3",D2(11+12)
  print *,"extent 3",D2(12+12)
  print *,"stride 3",D2(15+12)
end
