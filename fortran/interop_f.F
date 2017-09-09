module mytype
    use iso_c_binding
    implicit none
    type, bind(C) :: args_t
        integer(c_int) :: i
        integer(c_int) :: j
        integer(c_int) :: k
        real(c_double) :: x
        real(c_double) :: y
        real(c_double) :: z
    end type args_t
end module mytype

module mycall
    use iso_c_binding
    interface
        subroutine foo(a) bind(C, name="foo")
            use mytype
            type(args_t), value :: a
        end subroutine foo
    end interface
    interface
        subroutine bar(b) bind(C, name="bar")
            use mytype
            type(args_t) :: b
        end subroutine bar
    end interface
    interface
        subroutine zap(x) bind(C, name="bar")
            use iso_c_binding
            integer(c_long), allocatable :: x(:)
        end subroutine zap
    end interface
end module mycall

program test
    use iso_c_binding
    use mytype
    use mycall
    implicit none
    type(args_t) :: a
    type(args_t), allocatable :: b
    integer(c_long) :: i
    integer(c_long), allocatable :: x(:)
    a%i = 1
    a%j = 2
    a%k = 3
    a%x = 1.0
    a%y = 2.0
    a%z = 3.0
    call foo(a)
    allocate( b )
    b%i = 1
    b%j = 2
    b%k = 3
    b%x = 1.0
    b%y = 2.0
    b%z = 3.0
    call bar(b)
    deallocate( b )
    allocate( x(100), source=[(i,i=1,100)] )
    call zap(x)
    deallocate( x )
end program test


