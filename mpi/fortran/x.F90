program main
  use iso_fortran_env
  use mpi_f08
  implicit none
  type(MPI_Win) :: WA
  type(c_ptr) :: XA
  call MPI_Win_allocate(size=100, disp_unit=1, info=MPI_INFO_NULL, comm=MPI_COMM_WORLD, baseptr=XA, win=WA)
end program main

