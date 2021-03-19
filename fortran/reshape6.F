        program r
        implicit none
        integer :: a,b,c,d,e,f,k,errors
        real ::  X(17,15,13,11,7,5)
        real ::  Y(5,7,11,13,15,17)
        k = 1
        do a=1,17
         do b=1,15
          do c=1,13
           do d=1,11
            do e=1,7
             do f=1,5
              X(a,b,c,d,e,f) = k
              k = k + 1
             enddo
            enddo
           enddo
          enddo
         enddo
        enddo
        Y = 0
        Y = reshape(X, (/5,7,11,13,15,17/), order=(/6,5,4,3,2,1/))
        errors = 0
        do a=1,17
         do b=1,15
          do c=1,13
           do d=1,11
            do e=1,7
             do f=1,5
              print*,a,b,c,d,e,f,X(a,b,c,d,e,f),Y(f,e,d,c,b,a)
              if (X(a,b,c,d,e,f).ne.Y(f,e,d,c,b,a)) then
               errors = errors + 1
              endif
             enddo
            enddo
           enddo
          enddo
         enddo
        enddo
        print*,'ERRORS=',errors
        end program r
