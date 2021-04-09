program main
  use iso_fortran_env
  use omp_lib
  implicit none
  integer :: err
  ! problem definition
  integer(kind=INT32) ::  iterations
  integer(kind=INT64) ::  length
  real(kind=REAL64), allocatable ::  A(:)
  real(kind=REAL64), allocatable ::  B(:)
  real(kind=REAL64), allocatable ::  C(:)
  real(kind=REAL64) :: scalar
  integer(kind=INT64) :: bytes
  ! runtime variables
  integer(kind=INT64) :: i
  integer(kind=INT32) :: k
  real(kind=REAL64) ::  asum, ar, br, cr
  real(kind=REAL64) ::  t0, t1, nstream_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.D-8

  iterations = 100
  length = 1024*1024*128

  write(*,'(a,i12)') 'OpenMP default device = ', omp_get_default_device()
  write(*,'(a,i12)') 'Number of iterations  = ', iterations
  write(*,'(a,i12)') 'Matrix length         = ', length

  allocate( A(length), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of A returned ',err
    stop 1
  endif

  allocate( B(length), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of B returned ',err
    stop 1
  endif

  allocate( C(length), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of C returned ',err
    stop 1
  endif

  scalar = 3

  t0 = 0

  !$omp target data map(tofrom: A) map(to: B,C) map(to:length)

  !$omp target teams distribute simd
  do i=1,length
    A(i) = 0
    B(i) = 2
    C(i) = 2
  enddo
  !$omp end target teams distribute simd

  do k=0,iterations

    if (k.eq.1) t0 = omp_get_wtime()

    !$omp target teams distribute simd
    do i=1,length
      A(i) = A(i) + B(i) + scalar * C(i)
    enddo
    !$omp end target teams distribute simd

  enddo ! iterations

  t1 = omp_get_wtime()

  !$omp end target data

  nstream_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  ar  = 0
  br  = 2
  cr  = 2
  do k=0,iterations
      ar = ar + br + scalar * cr;
  enddo

  asum = 0
  !$omp parallel do reduction(+:asum)
  do i=1,length
    asum = asum + abs(A(i)-ar)
  enddo
  !$omp end parallel do

  deallocate( C )
  deallocate( B )
  deallocate( A )

  if (abs(asum) .gt. epsilon) then
    write(*,'(a35)') 'Failed Validation on output array'
    write(*,'(a30,f30.15)') '       Expected value: ', ar
    write(*,'(a30,f30.15)') '       Observed value: ', A(1)
    write(*,'(a35)')  'ERROR: solution did not validate'
    stop 1
  else
    write(*,'(a17)') 'Solution validates'
    avgtime = nstream_time/iterations;
    bytes = 4 * int(length,INT64) * storage_size(A)/8
    write(*,'(a12,f15.3,1x,a12,e15.6)')    &
        'Rate (MB/s): ', 1.d-6*bytes/avgtime, &
        'Avg time (s): ', avgtime
  endif

end program main

