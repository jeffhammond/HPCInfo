module cfi
    interface
        subroutine foo(buffer,m,n,o) bind(C,name="foo")
            implicit none
            class(*), dimension(*) :: buffer
            integer :: m, n, o
        end subroutine foo
    end interface

    interface
        subroutine bar(buffer,m,n, o) bind(C,name="bar")
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
    real(8), dimension(5,7,9) :: b
    integer :: i, j, k

    do i=1,size(a,3)
      do j=1,size(a,2)
        do k=1,size(a,1)
          a(k,j,i) = 10*k + 10*100*j + 10*100*100*i
          b(k,j,i) = 10.*k + 10.*100*j + 10.*100*100*i
        end do
      end do
    end do

    call foo(a,size(a,1),size(a,2),size(a,3))

    print*,'========================'
    call bar(a,size(a,1),size(a,2),size(a,3))

    print*,'========================'
    call bar(a(1:3,1:5,1:7),3,5,7)

    print*,'========================'
    call bar(a(2:4,2:6,2:8),3,5,7)

    !print*,'========================'
    !call bar(b,size(b,1),size(b,2),size(b,3))

end program test
