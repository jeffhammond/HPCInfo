      program swap
      implicit none
      integer :: ni,me
      real :: a(1000)[*]
      me=this_image()
      ni=num_images()
      a(1:1000)=me
      if ( me.eq.2 ) a(1:1000)=a(1:1000)[1]
      call sync_all()
      if (this_image().eq.2) then
        print*,'a(:) = ',a(:)
      end if
      end program swap
