      program atomic
      use iso_fortran_env
      use omp_lib
      implicit none
      integer :: i
      integer(atomic_int_kind) :: atom[*]
      call atomic_define (atom[1], this_image())
      !$OMP ATOMIC
      atom[1] = -this_image()
      end program atomic
