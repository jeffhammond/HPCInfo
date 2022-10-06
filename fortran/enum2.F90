       Program example
         Use enum_mod
         Type(myenum) :: x = one           ! Assign enumerator to enum-type var.
         Type(myenum) :: y = myenum(12345) ! Using the constructor.
         Type(myenum) :: x2 = myenum(two)  ! Constructor not needed but valid.
         Call sub(x)
         Call sub(three)
         Call sub(myenum(-Huge(one)))
       End Program example
