        program r
        implicit none
        integer :: a,b,c,k,errors
        real ::  X(13,11,7)
        real ::  Y(7,11,13)
        k = 1
        do a=1,13
         do b=1,11
          do c=1,7
           X(a,b,c) = k
           k = k + 1
          enddo
         enddo
        enddo
        Y = 0
        Y = reshape(X, (/7,11,13/), order=(/3,2,1/))
        errors = 0
        do a=1,13
         do b=1,11
          do c=1,7
           print*,a,b,c,X(a,b,c),Y(c,b,a)
           if (X(a,b,c).ne.Y(c,b,a)) then
            errors = errors + 1
           endif
          enddo
         enddo
        enddo
        print*,'ERRORS=',errors
        end program r
