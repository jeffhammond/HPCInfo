module pgif
    use, intrinsic :: ISO_C_binding
    use, intrinsic :: ISO_Fortran_env
    implicit none

    contains

        subroutine foo(b,m,n,p) bind(C,name="foo")
            real(kind=REAL64), dimension(m,n,p) :: b
            integer, value :: m, n, p
            integer :: i, j, k
            print*,'foo',m
            print*,'foo',n
            print*,'foo',p
            do k=1,p
             do j=1,n
              do i=1,m
                print*,'fooxxx',i,j,k,b(i,j,k)
              end do
             end do
            end do
        end subroutine foo

        subroutine bar(b,m,n,p) bind(C,name="bar")
            real(kind=REAL64), dimension(:,:,:) :: b
            integer, value :: m, n, p
            integer :: i, j, k
            print*,'bar',m
            print*,'bar',n
            print*,'bar',p
            print*,'bar CONTIGUOUS:',is_contiguous(b)
            print*,'bar S',size(b,1)
            print*,'bar S',size(b,2)
            print*,'bar S',size(b,3)
            print*,'bar L',lbound(b,1)
            print*,'bar L',lbound(b,2)
            print*,'bar L',lbound(b,3)
            print*,'bar U',ubound(b,1)
            print*,'bar U',ubound(b,2)
            print*,'bar U',ubound(b,3)
            do k=1,p
             do j=1,n
              do i=1,m
                print*,'barxxx',i,j,k,b(i,j,k)
              end do
             end do
            end do
        end subroutine bar

end module pgif

