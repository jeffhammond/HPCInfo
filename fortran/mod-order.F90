subroutine f(i)
  implicit none
  integer, intent(in) :: i
  print*,i 
end subroutine f

program main
  implicit none
  interface
    subroutine f(i)
      implicit none
      integer, intent(in) :: i
    end subroutine f
  end interface
  integer :: i
  i = 2
  call f(i)
end program main
