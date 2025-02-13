program main
  use, intrinsic :: iso_fortran_env
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr) :: P
  integer(kind=c_size_t), parameter :: page_size = 4096
  integer, parameter :: n = 1000
  integer(kind=c_size_t) :: bytes
  real(kind=REAL64), pointer :: B(:,:)

  interface

    subroutine free(ptr) bind(C,name="free")
      import c_ptr
      type(c_ptr), intent(in), value :: ptr
    end subroutine free

    !subroutine my_allocate(size, align, baseptr) bind(C,name="my_allocate")
    !  import c_ptr, c_size_t
    !  implicit none
    !  integer(kind=c_size_t), intent(in), value :: size, align
    !  type(c_ptr), intent(out) :: baseptr
    !end subroutine my_allocate

    ! void *malloc(size_t size);
    type(c_ptr) function malloc(size) bind(C,name="malloc")
      import c_ptr, c_size_t
      implicit none
      integer(kind=c_size_t), value :: size
    end function malloc    

    ! void *aligned_alloc(size_t alignment, size_t size);
    type(c_ptr) function aligned_alloc(alignment, size) bind(C,name="aligned_alloc")
      import c_ptr, c_size_t
      implicit none
      integer(kind=c_size_t), value :: alignment, size
    end function aligned_alloc    

    ! int posix_memalign(void **memptr, size_t alignment, size_t size);
    !integer(c_int) function posix_memalign(memptr, alignment, size) bind(C,name="posix_memalign")
    !  import c_ptr, c_size_t, c_int
    !  implicit none
    !  type(c_ptr) :: memptr
    !  integer(kind=c_size_t), value :: alignment, size
    !end function posix_memalign

  end interface

  bytes = n * n * 8 ! lazy: sizeof(REAL64) = 8

  !call my_allocate(size = bytes, align = page_size, baseptr = P)
  !P = malloc(size = bytes)
  P = aligned_alloc(alignment = page_size, size = bytes)

  call c_f_pointer(P,B,[n,n])

  B = 0
                        
  block
    integer :: i,j
    do i=1,n
      do j=1,n
        B(j,i) = i+j
      end do
    end do
  end block

  print*,B(1,1),B(n,n)

  call free(P)

end program main

