program r
    use omp_lib
    implicit none

    integer :: i,j,k

    real, allocatable, dimension(:,:) :: a,b,c

    integer, parameter :: big=1000000, small=10

    allocate( c(big,small), a(big,small), b(small,small) )

    a = 1
    b = 1
    c = 0

    !$omp parallel do reduction(+:c)
    do i=1,big
        do j=1,small
            do k=1,small
                c(i,j) = c(i,j) + a(i,k) * b(k,j)
            end do
        end do
    end do

    print*,c(1:big:big/10,1)
    print*,omp_get_max_threads()

    deallocate( a, b, c )

end program r
