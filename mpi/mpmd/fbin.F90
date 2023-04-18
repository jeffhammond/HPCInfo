program main
    use, intrinsic :: iso_fortran_env
    use, intrinsic :: iso_c_binding
    use omp_lib
    use mpi_f08
    implicit none
    integer :: required, provided
    required = MPI_THREAD_SERIALIZED
    call MPI_Init_thread(required,provided)
    if (provided.lt.required) stop

    call MPI_Finalize
end program main
