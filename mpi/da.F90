PROGRAM hello_world_mpi
! include 'mpif.h'
  use iso_fortran_env
  use mpi
  implicit none

  integer(kind=INT32) :: process_Rank, size_Of_Cluster, ierror, message_Item, tag
  integer(kind=INT32), parameter :: one = 1, zero = 0
  integer :: scattered_Data
  integer, dimension(4) :: distro_Array
  distro_Array = (/39, 72, 129, 42/)

  call MPI_INIT(ierror)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, size_Of_Cluster, ierror)
  call MPI_COMM_RANK(MPI_COMM_WORLD, process_Rank, ierror)

  call Print_Hello_World(process_Rank,size_Of_Cluster)

  call MPI_Scatter(distro_Array, one, MPI_INTEGER, scattered_Data, one, MPI_INTEGER, zero, MPI_COMM_WORLD, ierror);
  print *, "Process ", process_Rank, "received: ", scattered_Data

  IF(process_Rank == zero) THEN
    message_Item = 42
    call MPI_SEND(message_Item, one, MPI_INTEGER, one, one, MPI_COMM_WORLD, ierror)
    print *, "Sending message containing: ", message_Item
  ELSE IF(process_Rank == one) THEN
    call MPI_RECV(message_Item, one, MPI_INTEGER, zero, one, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
    print *, "Received message containing: ", message_Item
  END IF


  call MPI_FINALIZE(ierror)
contains
  subroutine Print_Hello_World(process_Rank, size_Of_Cluster)
    use iso_fortran_env
    integer(kind=INT32) :: process_Rank, size_Of_Cluster
    integer :: i
    !$acc kernels
    do i = 1, 3
    print *, 'Hello World from process: ', process_Rank, 'of ', size_Of_Cluster
    end do
    !$acc end kernels
  end subroutine Print_Hello_World
END PROGRAM
