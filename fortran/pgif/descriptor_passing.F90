subroutine inner(Z,n) bind(C,name="inner")
    implicit none
    integer :: n, i
    !real :: Z(*)
    real :: Z(:)
    print*,'Z: size=',size(Z)
    print*,'Z: shape=',shape(Z)
    print*,'Z: lbound=',lbound(Z)
    print*,'Z: ubound=',ubound(Z)
    print*,'Z: kind=',kind(Z)
    print*,'Z: loc=',loc(Z)
    do i = 1, n
        write(*,'(a9,i4,a2,d12.5)') 'inner: Z(',i,')=',Z(i)
    end do
end subroutine inner

subroutine outer(X,n) bind(C,name="outer")
    implicit none
    interface
        subroutine fake(Y,n,D) bind(C,name="inner")
            implicit none
            integer :: n
            real :: Y(*)
            integer(8) :: D(*)
        end subroutine fake
    end interface
    integer :: n, i
    real :: X(:)
    integer(8) :: D(16)
    !print*,'loc(X)=',loc(X)
    !do i = 1, size(X)
    !    write(*,'(a9,i4,a2,d12.5)') 'outer: X(',i,')=',X(i)
    !end do
    D(1)  = 35           ! tag (version)
    D(2)  =  1           ! rank
    D(3)  = 28           ! kind
    D(4)  =  4           ! len
    D(5)  =  0           ! flags
    D(6)  = size(X)      ! lsize
    D(7)  = size(X)      ! gsize
    D(8)  =  0           ! lbase
    D(9)  =  0           ! gbase
    D(10) =  0           ! unused
    D(11) =  1           ! dim[0].lbound
    D(12) = size(X)      ! dim[0].extent
    D(13) =  0           ! dim[0].sstride
    D(14) =  0           ! dim[0].soffset
    D(15) =  1           ! dim[0].lstride
    D(16) =  1 + size(X) ! dim[0].ubound
    call fake(X,n,D)
end subroutine outer

program main
    implicit none
    interface
        subroutine outer(X,n) bind(C,name="outer")
            implicit none
            integer :: n
            real :: X(:)
        end subroutine outer
    end interface
    real :: W(10)
    integer :: i
    do i = 1, size(W)
        W(i) = i
    end do
    call outer(W,size(W))
    print*,'The End'
end program main
