program main
    use iso_fortran_env
    real(REAL128) :: z
    z = 1
    do while (z .gt. 0)
        z = z / 2
        print*,z
    end do
end program main
