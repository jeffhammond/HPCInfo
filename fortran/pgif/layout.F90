program t
interface
  subroutine foo(array)
    real :: array(:,:,:)
    !pgi$ ignore_tkr(c) array
  end subroutine
end interface
real :: array(10,11,12)
call foo(array(2:10:2,1:11:2,2:12:2))
end program
subroutine foo(array, desc)
real array(*)
integer(8) desc(*)
print *,desc(1)
print *,"rank",desc(2)
print *,"offset",desc(8)
print *,"lbound 1",desc(11)
print *,"extent 1",desc(12)
print *,"stride 1",desc(15)
print *,"lbound 2",desc(11+6)
print *,"extent 2",desc(12+6)
print *,"stride 2",desc(15+6)
print *,"lbound 3",desc(11+12)
print *,"extent 3",desc(12+12)
print *,"stride 3",desc(15+12)
end


