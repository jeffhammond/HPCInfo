module mpi
    use iso_c_binding
    type(c_ptr), bind(C,name="evp") :: thing
    interface
        subroutine p() bind(C,name="p")
        end subroutine
    end interface
end module mpi

program main
    use mpi
    implicit none
    call p
    print*,LOC(thing)
end program main
