module pgif
    use, intrinsic :: ISO_C_binding
    use, intrinsic :: ISO_Fortran_env

    contains

        subroutine foo(b,m,n,p) bind(C,name="foo")
            implicit none
            real(kind=REAL64), dimension(m,n,p) :: b
            integer, value :: m, n, p
            integer :: i, j, k
            print*,'foo',m
            print*,'foo',n
            print*,'foo',p
            do k=1,p
             do j=1,n
              do i=1,m
                print*,'foo',i,j,k,b(i,j,k)
              end do
             end do
            end do
        end subroutine foo

        subroutine bar(b,m,n,p) bind(C,name="bar")
            implicit none
            real(kind=REAL64), dimension(:,:,:) :: b
            integer, value :: m, n, p
            integer :: i, j, k
            print*,'bar',m
            print*,'bar',n
            print*,'bar',p
            print*,'bar',size(b,1)
            print*,'bar',size(b,2)
            print*,'bar',size(b,3)
            do k=1,p
             do j=1,n
              do i=1,m
                print*,'foo',i,j,k,b(i,j,k)
              end do
             end do
            end do
        end subroutine bar

end module pgif

