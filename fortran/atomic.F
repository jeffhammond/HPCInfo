subroutine a(x,y)
    implicit none
    real(8), intent(inout) :: x(:)
    real(8), intent(out) :: y
    integer :: n, i
    n = size(x)
    y = 0
    do concurrent (i=0:n)
      !$acc atomic
      y = y + x(i)
    end do
end subroutine a

subroutine o(x,y)
    implicit none
    real(8), intent(inout) :: x(:)
    real(8), intent(out) :: y
    integer :: n, i
    n = size(x)
    y = 0
    do concurrent (i=0:n)
      !$acc atomic
      y = y + x(i)
    end do
end subroutine o

subroutine c(x,y)
    use cudadevice
    implicit none
    real(8), intent(inout) :: x(:)
    real(8), intent(out) :: y
    real(8) :: z
    integer :: n, i
    n = size(x)
    y = 0
    do concurrent (i=0:n)
      z = atomicadd(y, x(i))
    end do
end subroutine c


program bomb
    implicit none
     

end program bomb
