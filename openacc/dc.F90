#if W==1
module w
contains
pure subroutine bar(J,K,L,T,A,B)
  implicit none
  integer, intent(in) :: J, K, L
  real, intent(in) :: A(:)
  real, intent(out) :: B(:)
  real, intent(inout) :: T(:)
  T(K) = A(J)
  B(J) = T(L)
end subroutine bar
end module w
#endif

subroutine foo(N, A, B, T, K, L)
#if W==1
  use w
#endif
  implicit none
  integer, intent(in) :: N, K(N), L(N)
  real, intent(in) :: A(N)
  real, intent(out) :: B(N)
  real, intent(inout) :: T(N)
  integer :: J
#if V==1
  do concurrent (J=1:N)
#elif V==2
  !$acc kernels
  do J=1,N
#elif V==3
  !$acc parallel loop
  do J=1,N
#endif
#if W==1
    call bar(J,K(J),L(J),T,A,B)
#else
    ! During execution, K(J) and L(J) are both always 1.  So the store
    ! and load to/from T() always affect T(1) in each iteration.  Since
    ! T(1) is defined in each iteration before it is referenced, this
    ! program conforms with F2008 and F2018.
    T(K(J)) = A(J)
    B(J) = T(L(J)) ! must be A(J) whenever K(J)==L(J)
#endif
  end do
#if V==2
  !$acc end kernels
#endif
end subroutine foo
