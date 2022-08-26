module cfi
    interface
        subroutine foo(buffer,m,n,o) bind(C,name="foo")
            implicit none
            class(*), dimension(*) :: buffer
            integer :: m, n, o
        end subroutine foo
    end interface

    interface
        subroutine bar(buffer,m,n,o) bind(C,name="bar")
            implicit none
            class(*), dimension(..) :: buffer
            integer :: m, n, o
        end subroutine bar
    end interface
end module cfi 

program test
    use iso_c_binding
    use cfi
    implicit none
    integer(4), dimension(5,7,9) :: a
    integer(4), dimension(:,:,:), allocatable :: c
    integer :: i, j, k

    allocate( c(5,7,9) )

    do i=1,size(a,3)
      do j=1,size(a,2)
        do k=1,size(a,1)
          a(k,j,i) = 10*k + 10*100*j + 10*100*100*i
        end do
      end do
    end do
    c = a

    call foo(a,size(a,1),size(a,2),size(a,3))
    call bar(a,size(a,1),size(a,2),size(a,3))
    call bar(c,size(c,1),size(c,2),size(c,3))

end program test
