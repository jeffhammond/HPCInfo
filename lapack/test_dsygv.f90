program test

    use iso_fortran_env
    use omp_lib

    implicit none

    ! for argument parsing
    integer :: err
    integer :: arglen
    character(len=32) :: argtmp

    ! timing repetitions
    integer :: reps = 20

    ! matrix dimension
    integer :: n
    ! matrix input, output and work arrays
    double precision, allocatable ::  a0(:,:), a(:,:) ! n,n
    double precision, allocatable ::  b0(:,:), b(:,:) ! n,n
    double precision, allocatable ::  e(:) ! n
    double precision, allocatable ::  w(:) ! 3*n

    double precision :: t0, t1, dt, esum

    ! LAPACK error code
    integer :: info

    ! loop indices
    integer :: i,j,k

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
    allocate( a0(n,n), stat=err)
    if (err .ne. 0) then
        write(*,'(a,i3)') 'allocation of A0 returned ',err
        stop
    endif
    allocate( a(n,n), stat=err)
    if (err .ne. 0) then
        write(*,'(a,i3)') 'allocation of A returned ',err
        stop
    endif
    allocate( b0(n,n), stat=err)
    if (err .ne. 0) then
        write(*,'(a,i3)') 'allocation of B0 returned ',err
        stop
    endif
    allocate( b(n,n), stat=err)
    if (err .ne. 0) then
        write(*,'(a,i3)') 'allocation of B returned ',err
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
    call random_number(a0)
    do j=1,n
        do i=1,j-1
            a0(j,i) = a0(i,j)
        enddo
    enddo

    ! initialize B
    call random_number(b0)
    do j=1,n
        do i=1,j-1
            b0(j,i) = b0(i,j)
        enddo
    enddo

    if (n.lt.100) then
        print*,'A=',a0
        print*,'B=',b0
    endif

    ! UPLO should not matter
    do k=0,reps
        a = a0
        b = b0
        t0 = omp_get_wtime()
        call dsygv(1,'V','U',n,a,n,b,n,e,w,3*n,info)
        if (info.ne.0) then
            if (info.lt.0) then
                print*,'argument ',-info,' is wrong'
            else if (info.gt.0) then
                print*,'DPOTRF or DSYEV returned an error code'
                if (info.le.n) then
                    print*,'DSYEV failed to converge'
                else
                    print*,'minor of B is not positive definite.'
                endif
            endif
            stop
        endif
        t1 = omp_get_wtime()
        if (k.ge.1) dt = dt + (t1-t0)
    enddo

    if (n.lt.100) then
        print*,'A=',a
        print*,'B=',b
        print*,'lambda=',e
    endif
    print*,'dt=',dt/reps

    esum = 0.0d0
    do i=1,n
        esum = esum + e(i)
    enddo
    print*,'esum=',esum

end program test
