      program ieee_test
        use iso_fortran_env
        use ieee_arithmetic
        real(kind=REAL64) :: x(1000)
        integer :: i
        logical :: bad
        bad = .false.
!$acc parallel loop reduction(iand:bad)
        do i=1,1000
            if (ieee_is_nan(x(i))) then
               bad = .true. 
            endif
        end do
        print*,bad
      end program ieee_test
