       Module enum_mod
         Enum,Bind(C) :: myenum
           Enumerator :: one=1, two, three
         End Enum
         Enum,Bind(C) :: flags
           Enumerator :: f1 = 1, f2 = 2, f3 = 4
         End Enum
       Contains
         Subroutine sub(a) Bind(C)
           Type(myenum),Value :: a
           Print *,a ! Prints the integer value, as if it were Print *,Int(a).
         End Subroutine
       End Module enum_mod
