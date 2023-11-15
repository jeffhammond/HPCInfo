program main
  use, intrinsic :: iso_fortran_env
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr) :: P
  integer(kind=c_size_t), parameter :: page_size = 4096
  integer, parameter :: n = 100
  integer(kind=c_size_t) :: bytes
  real(kind=REAL64), pointer :: B(:,:)

  interface
    ! void *aligned_alloc(size_t alignment, size_t size);
    type(c_ptr) function aligned_alloc(alignment, size) bind(C,name="aligned_alloc")
      import c_ptr, c_size_t
      implicit none
      integer(kind=c_size_t) :: alignment, size
    end function aligned_alloc    
#if 0
    ! int posix_memalign(void **memptr, size_t alignment, size_t size);
    integer(c_int) function posix_memalign(memptr, alignment, size) bind(C,name="posix_memalign")
      import c_ptr, c_size_t, c_int
      implicit none
      type(c_ptr) :: memptr
      integer(kind=c_size_t) :: alignment, size
    end function posix_memalign
#endif
  end interface

  bytes = n * n * 8 ! lazy: sizeof(REAL64) = 8
  P = aligned_alloc(alignment = page_size, size = bytes)
  call c_f_pointer(P,B,[n,n])
                        
  block
    integer :: i,j
    do i=1,n
      do j=1,n
        B(j,i) = i+j
      end do
    end do
  end block

end program main

