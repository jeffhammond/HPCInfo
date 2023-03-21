! USE mpi_f08
! MPI_Win_allocate(size, disp_unit, info, comm, baseptr, win, ierror)
!    USE, INTRINSIC :: ISO_C_BINDING, ONLY : C_PTR
!    INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) :: size
!    INTEGER, INTENT(IN) :: disp_unit
!    TYPE(MPI_Info), INTENT(IN) :: info
!    TYPE(MPI_Comm), INTENT(IN) :: comm
!    TYPE(C_PTR), INTENT(OUT) :: baseptr
!    TYPE(MPI_Win), INTENT(OUT) :: win
!    INTEGER, OPTIONAL, INTENT(OUT) :: ierror
program main
  use iso_fortran_env
  use mpi_f08
  implicit none
  integer(kind=MPI_ADDRESS_KIND) :: as = 100
  type(c_ptr) :: XA
  type(MPI_Win) :: WA
  TYPE(MPI_Comm) :: comm = MPI_COMM_WORLD
  call MPI_Win_allocate(as, 1, MPI_INFO_NULL, comm, XA, WA)
  call MPI_Win_allocate(100, 1, MPI_INFO_NULL, comm, XA, WA)
end program main

