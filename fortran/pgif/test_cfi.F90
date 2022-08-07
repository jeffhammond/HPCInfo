module cfi
    interface
        subroutine foo(buffer,m,n) bind(C,name="foo")
            implicit none
            integer, dimension(*) :: buffer
            integer :: m, n
        end subroutine foo
    end interface

    interface
        subroutine bar(buffer,m,n) bind(C,name="bar")
            implicit none
            integer, dimension(..) :: buffer
            integer :: m, n
        end subroutine bar
    end interface
end module cfi 

program test
    use iso_c_binding
    use cfi
    implicit none
    integer, dimension(10,10) :: a
    integer :: i, j

    do i=1,10
      do j=1,10
        a(j,i) = 100*j + 100*100*i
      end do
    end do

    call foo(a,size(a,1),size(a,2))
    !call foo(a(1,1),size(a,1),size(a,2))

    call bar(a,size(a,1),size(a,2))
    !call bar(a(1,1),size(a,1),size(a,2))

    !call foo(a(:,:))

    !call foo(a(:,1:5))
    !call foo(a(:,6:10))

    !call foo(a(1:5,:))
    !call foo(a(6:10,:))

    !call foo(a(1:5,1:5))
    !call foo(a(6:10,6:10))

end program test
