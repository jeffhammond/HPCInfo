      program main
      implicit none
      integer, allocatable :: A(:)[:]
      integer :: B
      if (num_images()<2) call abort;
      ! allocate from the shared heap
      allocate(A(1)[*])
      B = 37;
      ! store contents of local B at PE 0 into A at PE 1
      if (this_image().eq.0) A(1)[1] = B;
      ! global synchronization of execution and data
      sync all
      ! observe the result of the store
      if (this_image().eq.1) print*,'A@1=',A(1)[1]
      deallocate(A)
      end program main
