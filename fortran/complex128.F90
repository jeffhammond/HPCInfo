program main
    use iso_fortran_env
    complex(REAL128) :: z
    !$omp atomic write
    z = (0.0d0,0.0d0)
    !$omp target map(tofrom:z)
    !$omp teams
    !$omp parallel
    !$omp atomic update
    z = z + (1,1)
    !$omp end parallel
    !$omp end teams
    !$omp end target
    print*,z
end program main
