module m

contains

pure subroutine inner(x,y)
    implicit none
!$acc routine
    real, intent(in) :: x(:)
    real, intent(out) :: y(:)
    integer :: i, n
    n = max(size(x),size(y))
!$acc parallel loop worker
    do concurrent (i=1:n)
        y(i) = x(i)
    end do
end subroutine inner

end module m

subroutine outer(x,y)
    use m
    implicit none
    real, intent(in) :: x(:,:)
    real, intent(out) :: y(:,:)
    integer :: i, n
    n = max(size(x,2),size(y,2))
!$acc parallel loop gang
    do concurrent (i=1:n)
        call inner(x(:,i),y(:,i))
    end do
end subroutine outer
