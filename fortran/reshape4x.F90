#define D1 4
#define D2 4
#define D3 4
#define D4 4
program r
implicit none
integer :: a,b,c,d,errors
real ::  X(D1,D2,D3,D4)
real ::  Y(D4,D3,D2,D1)
integer :: p(4),q(4)
p = [4,3,2,1]
do d=1,size(X,4)
 do c=1,size(X,3)
  do b=1,size(X,2)
   do a=1,size(X,1)
    X(a,b,c,d) = d + 100*c + 10000*b + 1000000*a
   enddo
  enddo
 enddo
enddo
Y = 0
! the shape argument matches the output array
Y = reshape(X, [size(Y,1),size(Y,2),size(Y,3),size(Y,4)], order=p)
errors = 0
do d=1,size(X,4)
 do c=1,size(X,3)
  do b=1,size(X,2)
   do a=1,size(X,1)
    q = [a,b,c,d]
    q = [q(p(1)),q(p(2)),q(p(3)),q(p(4))]
    if (X(a,b,c,d).ne.Y(q(1),q(2),q(3),q(4))) then
      errors = errors + 1
      print*,a,b,c,d,X(a,b,c,d),q(1),q(2),q(3),q(4),Y(q(1),q(2),q(3),q(4)),'<<<<<'
    else
      print*,a,b,c,d,X(a,b,c,d),q(1),q(2),q(3),q(4),Y(q(1),q(2),q(3),q(4))
    endif
   enddo
  enddo
 enddo
enddo
print*,'ERRORS=',errors
end program r
