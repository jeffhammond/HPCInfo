module pgif
    use, intrinsic :: ISO_C_binding
    use, intrinsic :: ISO_Fortran_env

    contains

        subroutine foo(b,m,n) bind(C,name="foo")
            implicit none
            real(kind=REAL64), dimension(m,n) :: b
            integer, value :: m, n
            integer :: i, j
            print*,'foo',m
            print*,'foo',n
            !print*,'foo',b(1:m,1:n)
            do j=1,n
              do i=1,m
                print*,'foo',i,j,b(i,j)
              end do
            end do
        end subroutine foo

        subroutine bar(b,m,n) bind(C,name="bar")
            implicit none
            real(kind=REAL64), dimension(:,:) :: b
            integer, value :: m, n
            integer :: i, j
            print*,'bar',m
            print*,'bar',n
            print*,'bar',size(b,1)
            print*,'bar',size(b,2)
            !print*,'bar',b
            do j=1,n
              do i=1,m
                print*,'foo',i,j,b(i,j)
              end do
            end do
        end subroutine bar

end module pgif

