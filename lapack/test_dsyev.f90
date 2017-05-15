program test

    use ISO_FORTRAN_ENV
    use OMP_LIB

    implicit none

    ! for argument parsing
    integer :: err
    integer :: arglen
    character(len=32) :: argtmp

    ! matrix dimension
    integer :: n
    ! matrix input, output and work arrays
    double precision, allocatable ::  a(:,:) ! n,n
    double precision, allocatable ::  e(:) ! n
    double precision, allocatable ::  w(:) ! 3*n

    ! LAPACK error code
    integer :: info

    ! loop indices
    integer :: i,j,k

    ! temporary - to be removed
    double precision :: x

    call random_seed()

    ! parse matrix dimension argument
    n = 4
    call get_command_argument(1,argtmp,arglen,err)
    if (err.eq.0) read(argtmp,'(i32)') n
    if (n .lt. 1) then
        write(*,'(a,i5)') 'ERROR: n must be a positive number : ', n
        stop
    endif

    ! allocate arrays
    allocate( a(n,n), stat=err)
    if (err .ne. 0) then
        write(*,'(a,i3)') 'allocation of A returned ',err
        stop
    endif
    allocate( e(n), stat=err)
    if (err .ne. 0) then
        write(*,'(a,i3)') 'allocation of E returned ',err
        stop
    endif
    allocate( w(3*n), stat=err)
    if (err .ne. 0) then
        write(*,'(a,i3)') 'allocation of W returned ',err
        stop
    endif

    ! initialize A
    do j=1,n
        call random_number(x)
        x = x-0.5
        a(j,j) = x
        do i=1,j-1
            call random_number(x)
            x = x-0.5
            a(i,j) = x
            a(j,i) = x
        enddo
    enddo

    print*,'A=',a

    ! UPLO should not matter
    call dsyev('V','U',n,a,n,e,w,3*n,info)

    print*,'A=',a
    print*,'lambda=',e

end program test
