module affinity
    use, intrinsic :: iso_c_binding

    interface
        subroutine print_affinity() &
                   bind(C,name="print_affinity")
        end subroutine print_affinity
    end interface

end module affinity

program main
    use, intrinsic :: iso_fortran_env
    use, intrinsic :: iso_c_binding
    use omp_lib
    use mpi_f08
    use affinity
    implicit none
    integer :: me, np

    block
        integer :: required, provided
        required = MPI_THREAD_SERIALIZED
        call MPI_Init_thread(required,provided)
        if (provided.lt.required) stop
    end block

    block
        call MPI_Comm_rank(MPI_COMM_WORLD, me)
        call MPI_Comm_size(MPI_COMM_WORLD, np)
        write(6,'(a,i3,a,i3)') 'F: I am ',me,' of ',np
    end block

    flush(0)
    call MPI_Barrier(MPI_COMM_WORLD)

    block
        type(MPI_Comm) :: node
        integer :: node_me, node_np
        call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, node)
        call MPI_Comm_rank(node, node_me)
        call MPI_Comm_size(node, node_np)
        write(6,'(a,i3,a,i3,a,i3)') 'F: rank ',me,' is the ',node_me,' rank of ',node_np
    end block

    flush(0)
    call MPI_Barrier(MPI_COMM_WORLD)

    call MPI_Finalize
end program main
