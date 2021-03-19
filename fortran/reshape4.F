        program r
        implicit none
        integer :: a,b,c,d,k,errors
        real ::  X(13,11,7,5)
        real ::  Y(5,7,11,13)
        k = 1
        do a=1,13
         do b=1,11
          do c=1,7
           do d=1,5
            X(a,b,c,d) = k
            k = k + 1
           enddo
          enddo
         enddo
        enddo
        Y = 0
        Y = reshape(X, (/5,7,11,13/), order=(/4,3,2,1/))
        errors = 0
        do a=1,13
         do b=1,11
          do c=1,7
           do d=1,5
            print*,a,b,c,d,X(a,b,c,d),Y(d,c,b,a)
            if (X(a,b,c,d).ne.Y(d,c,b,a)) then
              errors = errors + 1
            endif
           enddo
          enddo
         enddo
        enddo
        print*,'ERRORS=',errors
        end program r
