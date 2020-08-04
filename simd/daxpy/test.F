program test
  use iso_fortran_env
#ifdef _OPENMP
  use omp_lib
#endif
  implicit none
  double precision, allocatable :: x(:)
  double precision, allocatable :: y(:)
  double precision, parameter :: alpha = 1.0d0, beta = 1.0d0
  integer :: iterations, length, offset, bytes
  integer, parameter :: one = 1
  integer :: me, nt, mt
  double precision :: t0,t1,dt
  integer :: i
  ! for argument parsing
  integer :: err
  integer :: cac
  integer :: arglen
  character(len=32) :: argtmp

  cac = command_argument_count()
  if (cac .lt. 2) then
    write(*,'(a17,i1)') 'argument count = ', cac
    write(*,'(a33)')    'Usage: ./test.x iterations length'
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    write(*,'(a10,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  length = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') length
  if (length .lt. 1) then
    write(*,'(a10,i5)') 'ERROR: length must be nonnegative : ', length
    stop 1
  endif

  if ( size(x) .ne. size(y) ) then
    write(*,'(a10)') 'x and y are not the same size!'
  endif

  mt = omp_get_max_threads()
  if ( size(x) .le. ( mt * ceiling( size(x) / 4096.0d0 ) ) ) then
    write(*,'(a10)') 'need more padding in x/y arrays'
  endif

  write(*,'(a10)')  'DAXPY test:'
  write(*,'(a10,i10)')  'iterations  = ', iterations
  write(*,'(a10,i10)')  'length      = ', length
  write(*,'(a10,i10)')  'max threads = ', mt


  allocate( x(length*mt), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of x returned ',err
    stop 1
  endif

  allocate( y(length*mt), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of y returned ',err
    stop 1
  endif

  !$omp parallel private(me,nt,offset,t0,t1,dt)

  me = omp_get_thread_num()
  nt = omp_get_num_threads()

  offset = me * ceiling((1.0d0*size(x)) / (1.0d0*nt))

  do i = 1, length
    x(offset+i) = 3.0d0
    y(offset+i) = 0.0d0
  enddo

  t0 = 0.0d0
  do i = 0, iterations
    if (i.eq.1) then
      !$omp barrier
      t0 = omp_get_wtime()
    endif
    call daxpy(length,alpha,x(offset+1),one,y(offset+1),one)
  enddo
  !$omp barrier
  t1 = omp_get_wtime()

  do i = 1, length
    if ( y(offset+i) .ne. (3.0d0 * (iterations+1)) ) then
      write(*,'(a10,i3,a5,i10,a4,f8.3,a4,f30.15)') 'ERROR: me : ', me,' off=', offset+i, ' ref=', 3.0d0 * iterations,' val=', y(offset+i)
    endif
  enddo

  dt = (t1 - t0) / iterations
  ! load X, load Y, store Y => 3
  bytes = 3 * length  * storage_size(x)/8
  !$omp critical
  write(*,'(a7,i3)')         'thread=',me
  write(*,'(a5,f30.15)')     'time=',dt
  write(*,'(a6,i10)')        'bytes=',bytes
  write(*,'(a10,f30.15,a5)') 'bandwidth=',(1.0d-9)*bytes/dt,' GB/s'
  !$omp end critical

  !$omp end parallel

end program test
