module m

contains

pure subroutine inner(x,y)
    implicit none
!$acc routine worker
    real, intent(in) :: x(:)
    real, intent(out) :: y(:)
    integer :: i, n
    n = max(size(x),size(y))
!$acc loop
    do concurrent (i=1:n)
        y(i) = x(i)
    end do
end subroutine inner

subroutine outer(x,y)
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

end module m

program main
    use m
    implicit none
    real :: x(1024,1024)
    real :: y(1024,1024)
    x = 1
    y = 0
    call outer(x,y)
    if (any(x.ne.y)) then
        print*,'bad'
    else
        print*,'good'
    end if
end program main
