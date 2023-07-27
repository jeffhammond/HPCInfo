program r
    implicit none

    integer :: i,j,k

    real, allocatable, dimension(:,:) :: a,b,c

    allocate( c(100000000,10), a(100000000,10), b(10,10) )

    a = 1
    b = 1
    c = 0

    !$omp parallel do collapse(3)
    do i=1,100000000
        do j=1,10
            do k=1,10
                !$omp atomic update
                c(i,j) = c(i,j) + a(i,k) * b(k,j)
                !$omp end atomic
            end do
        end do
    end do

    print*,c(1:100000000:10000000,1)

    deallocate( a, b, c )

end program r
