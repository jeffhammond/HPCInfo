program main
  use iso_fortran_env
  use mpi_f08
  implicit none
  integer(kind=INT32) ::  iterations, order, block_order
  type(MPI_Win) :: WA                               ! MPI window for A (original matrix)
  type(c_ptr) :: XA                                 ! MPI baseptr / C pointer for A
  real(kind=REAL64), pointer     ::  A(:,:)         ! Fortran pointer to A
  real(kind=REAL64), parameter :: one=1.0d0
  integer(kind=INT32) :: me, np, provided
  integer(kind=INT32) :: dsize

  call MPI_Init_thread(MPI_THREAD_SINGLE,provided)
  call MPI_Comm_rank(MPI_COMM_WORLD, me)
  call MPI_Comm_size(MPI_COMM_WORLD, np)

  iterations = 1
  order = 5*7*8*9
  block_order = int(order / np)

  call MPI_Barrier(MPI_COMM_WORLD)

  dsize = storage_size(one)/8
  ! MPI_Win_allocate(size, disp_unit, info, comm, baseptr, win, ierror)
  call MPI_Win_allocate(size=block_order * order * dsize, disp_unit=dsize, &
                        info=MPI_INFO_NULL, comm=MPI_COMM_WORLD, baseptr=XA, win=WA)
  call MPI_Win_lock_all(0,WA)

  call c_f_pointer(XA,A,[block_order,order])
                        
  call MPI_Win_unlock_all(WA)
  call MPI_Win_free( WA)
  call MPI_Finalize()
end program main

