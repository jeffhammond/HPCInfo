#define D1 9
#define D2 7
#define D3 5
#define D4 3
program test_reshape
implicit none
integer :: a,b,c,d,errors
real, allocatable, dimension(:,:,:,:) ::  X,Y
integer, dimension(4) :: p,q,r,s
s = [D1,D2,D3,D4]
!p = [1,2,3,4]
!p = [1,2,4,3]
!p = [1,3,2,4]
!p = [1,3,4,2]
!p = [1,4,2,3]
!p = [1,4,3,2]
!p = [2,1,4,3]
!p = [2,1,3,4]
!p = [2,3,1,4]
!p = [2,3,4,1]
!p = [2,4,1,3]
!p = [2,4,3,1]
!p = [3,1,2,4]
!p = [3,1,4,2]
!p = [3,2,1,4]
!p = [3,2,4,1]
!p = [3,4,1,2]
!p = [3,4,2,1]
!p = [4,1,2,3]
!p = [4,1,3,2]
!p = [4,2,1,3]
!p = [4,2,3,1]
p = [4,3,1,2]
!p = [4,3,2,1]
print*,'permutation=',p
r=[s(p(1)),s(p(2)),s(p(3)),s(p(4))]
allocate( X(s(1),s(2),s(3),s(4)) , Y(r(1),r(2),r(3),r(4)) )
print*,'shape(X)=',shape(X)
print*,'shape(Y)=',shape(Y)
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
Y = reshape(X, shape(Y), order=p)
errors = 0
do a=1,size(X,1)
 do b=1,size(X,2)
  do c=1,size(X,3)
   do d=1,size(X,4)
    write(*,'(a3,4i3,f12.0)') 'X',a,b,c,d,X(a,b,c,d)
   enddo
  enddo
 enddo
enddo
do a=1,size(Y,1)
 do b=1,size(Y,2)
  do c=1,size(Y,3)
   do d=1,size(Y,4)
    write(*,'(a3,4i3,f12.0)') 'Y',a,b,c,d,Y(a,b,c,d)
   enddo
  enddo
 enddo
enddo
print*,'ERRORS=',errors
end program test_reshape
