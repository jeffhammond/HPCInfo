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
        !integer(4), dimension(-2:2,-3:3,-4:4) :: q0
        !integer(4), dimension(-1:3,-2:4,-3:5) :: q1
        !integer(4), dimension( 0:4,-1:5,-2:6) :: q2
        !integer(4), dimension( 1:5, 0:6,-1:7) :: q3
        !integer(4), dimension( 2:6, 1:7, 0:8) :: q4
        !integer(4), dimension( 3:7, 2:8, 1:9) :: q5
        integer(4), dimension( 0:2, 0:3, 0:5) :: q6
        !print*,'================================'
        !call bar(q0)
        !print*,'================================'
        !call bar(q1)
        !print*,'================================'
        !call bar(q2)
        !print*,'================================'
        !call bar(q3)
        !print*,'================================'
        !call bar(q4)
        !print*,'================================'
        !call bar(q5)
        print*,'================================'
        call bar(q6)
    end block

end program test
