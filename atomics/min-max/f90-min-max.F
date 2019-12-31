program main
  use iso_fortran_env
  use omp_lib
  implicit none

  integer, parameter :: iterations = 10000
  integer, parameter :: elements = 1024

  integer(kind=INT32) :: i32(elements)
  integer(kind=INT64) :: i64(elements)
  real(kind=REAL32)   :: r32(elements)
  real(kind=REAL64)   :: r64(elements)

  integer(kind=INT32) :: i32_min
  integer(kind=INT64) :: i64_min
  real(kind=REAL32)   :: r32_min
  real(kind=REAL64)   :: r64_min

  integer(kind=INT32) :: i32_max
  integer(kind=INT64) :: i64_max
  real(kind=REAL32)   :: r32_max
  real(kind=REAL64)   :: r64_max

  integer :: i,j,k

  integer :: me, nt
  integer :: start, stop, chunk

  real(kind=REAL64)   :: t0, t1, dt

  do j=1,elements
    k = MOD(j,37)
    i32(j) = k
    i64(j) = k
    r32(j) = real(k,REAL32)
    r64(j) = real(k,REAL64)
  enddo

  i32_min =  100000
  i64_min =  100000
  r32_min =  100000
  r64_min =  100000

  i32_max = -100000
  i64_max = -100000
  r32_max = -100000
  r64_max = -100000

  !$omp parallel private(me,nt,chunk,start,stop)

  me    = omp_get_thread_num()
  nt    = omp_get_num_threads()
  chunk = elements/nt
  if (MOD(elements,nt).ne.0) chunk = chunk + 1
  start = chunk*me
  stop  = chunk*(me+1)
  if (stop .gt. elements) stop = elements

  !$omp critical
  print*,'me,nt,chunk,start,stop=',me,nt,chunk,start,stop
  !$omp end critical

  ! MIN

  ! warmup
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      i32_min = MIN(i32_min,i32(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t0 = omp_get_wtime()
  !$omp end master

  ! timed
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      i32_min = MIN(i32_min,i32(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t1 = omp_get_wtime()
  dt = t1 - t0
  write(*,'(a4,1x,a4,i10,1x,a4,1x,f12.7,1x,a7)') 'i32:','min=',i32_min,'took',dt,'seconds'
  !$omp end master

  ! warmup
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      i64_min = MIN(i64_min,i64(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t0 = omp_get_wtime()
  !$omp end master

  ! timed
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      i64_min = MIN(i64_min,i64(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t1 = omp_get_wtime()
  dt = t1 - t0
  write(*,'(a4,1x,a4,i10,1x,a4,1x,f12.7,1x,a7)') 'i64:','min=',i64_min,'took',dt,'seconds'
  !$omp end master

  ! warmup
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      r32_min = MIN(r32_min,r32(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t0 = omp_get_wtime()
  !$omp end master

  ! timed
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      r32_min = MIN(r32_min,r32(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t1 = omp_get_wtime()
  dt = t1 - t0
  write(*,'(a4,1x,a4,e10.3,1x,a4,1x,f12.7,1x,a7)') 'r32:','min=',r32_min,'took',dt,'seconds'
  !$omp end master

  ! warmup
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      r64_min = MIN(r64_min,r64(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t0 = omp_get_wtime()
  !$omp end master

  ! timed
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      r64_min = MIN(r64_min,r64(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t1 = omp_get_wtime()
  dt = t1 - t0
  write(*,'(a4,1x,a4,e10.3,1x,a4,1x,f12.7,1x,a7)') 'r64:','min=',r64_min,'took',dt,'seconds'
  !$omp end master

  ! MAX

  ! warmup
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      i32_max = MAX(i32_max,i32(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t0 = omp_get_wtime()
  !$omp end master

  ! timed
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      i32_max = MAX(i32_max,i32(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t1 = omp_get_wtime()
  dt = t1 - t0
  write(*,'(a4,1x,a4,i10,1x,a4,1x,f12.7,1x,a7)') 'i32:','max=',i32_max,'took',dt,'seconds'
  !$omp end master

  ! warmup
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      i64_max = MAX(i64_max,i64(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t0 = omp_get_wtime()
  !$omp end master

  ! timed
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      i64_max = MAX(i64_max,i64(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t1 = omp_get_wtime()
  dt = t1 - t0
  write(*,'(a4,1x,a4,i10,1x,a4,1x,f12.7,1x,a7)') 'i64:','max=',i64_max,'took',dt,'seconds'
  !$omp end master

  ! warmup
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      r32_max = MAX(r32_max,r32(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t0 = omp_get_wtime()
  !$omp end master

  ! timed
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      r32_max = MAX(r32_max,r32(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t1 = omp_get_wtime()
  dt = t1 - t0
  write(*,'(a4,1x,a4,e10.3,1x,a4,1x,f12.7,1x,a7)') 'r32:','max=',r32_max,'took',dt,'seconds'
  !$omp end master

  ! warmup
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      r64_max = MAX(r64_max,r64(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t0 = omp_get_wtime()
  !$omp end master

  ! timed
  do i=1,iterations
    do j=1,elements
      !$omp atomic update
      r64_max = MAX(r64_max,r64(j))
      !$omp end atomic
    enddo
  enddo

  !$omp barrier
  !$omp master
  t1 = omp_get_wtime()
  dt = t1 - t0
  write(*,'(a4,1x,a4,e10.3,1x,a4,1x,f12.7,1x,a7)') 'r64:','max=',r64_max,'took',dt,'seconds'
  !$omp end master

  !$omp end parallel


end program main
