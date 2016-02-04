program ieee_test
use iso_fortran_env
use ieee_arithmetic
real(kind=REAL64) :: x
if (ieee_support_datatype(x)) then
    stop 1
else
    stop 0
endif
end program ieee_test
