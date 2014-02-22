      program atomic
      implicit none
      integer :: i[*]
      integer :: n
      i = 0
      critical
        i[1] = i[1] + this_image()
      end critical
      ! sync all ! also works
      call sync_all()
      if (this_image().eq.1) then
        n = num_images()
        print*,n,' images'
        print*,'i[1] = ',i[1]
        print*,'n*(n+1)/2',n*(n+1)/2
      end if
      end program atomic
