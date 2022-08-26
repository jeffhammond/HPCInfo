module cfi
    interface
        subroutine foo(buffer,dims,ndim) bind(C,name="foo")
            implicit none
            class(*), dimension(*) :: buffer
            integer, dimension(*) :: dims
            integer, value :: ndim
        end subroutine foo
    end interface

    interface
        subroutine bar(buffer) bind(C,name="bar")
            implicit none
            class(*), dimension(..) :: buffer
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

    call foo(a,shape(a),size(shape(a)))
    print*,'================================'
    call bar(a)
    print*,'================================'
    call bar(c)

    block
        integer(4), dimension(-2:2,-3:3,-4:-4) :: q
        q = -1
        call bar(q)
    end block

end program test
