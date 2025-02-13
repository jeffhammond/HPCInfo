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

subroutine foo(N, A, B, T, K, L)
  use w
  implicit none
  integer, intent(in) :: N, K(N), L(N)
  real, intent(in) :: A(N)
  real, intent(out) :: B(N)
  real, intent(inout) :: T(N)
  do concurrent (integer :: J=1:N)
    call bar(J,K(J),L(J),T,A,B)
  end do
end subroutine foo
