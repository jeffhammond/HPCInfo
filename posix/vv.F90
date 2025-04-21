module hwcount

    interface
        function get_num_cpus_from_mask() result(num_cpus) &
                 bind(C,name="get_num_cpus_from_mask")
            use iso_c_binding, only: c_int
            implicit none
            integer(kind=c_int) :: num_cpus
        end function get_num_cpus_from_mask
    end interface

    contains

    subroutine oversubscription_warning(print_mask)
        implicit none
        logical, intent(in) :: print_mask
        integer :: num_cpus
        num_cpus = get_num_cpus_from_mask()
        if (print_mask) then
          if (num_cpus .le. 0) then
            write(*,'(a)') 'get_num_cpus_from_mask failed'
          else if  (num_cpus .le. 2) then
            write(*,'(a)') 'Performance warning'
          else
            write(*,'(a)') 'Everything is fine'
          end if
        end if
    end subroutine oversubscription_warning

end module hwcount

#if TEST
program test
    use mpi
    use hwcount
    implicit none
    integer :: me, ierror
    call MPI_Init(ierror)
    call MPI_Comm_rank(MPI_COMM_WORLD,me,ierror)
    call oversubscription_warning(me.eq.0)
    call MPI_Finalize(ierror)
end program test
#endif
