program main
    implicit none
    integer i
    !$omp parallel
    !$omp do   
    do i = 1, 1000
        print*,'loop 1 ',i
    enddo
    !$omp do   
    do i = 1, 1000
        print*,'loop 2 ',i
    enddo
    !$omp sections
    !$omp section
    print*,'section 1'
    !$omp section
    print*,'section 2'
    !$omp end sections
    !$omp end parallel
end program main
