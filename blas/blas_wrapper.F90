module blas

    use iso_fortran_env, only: INT32, INT64

    implicit none

! BLAS will be using 32-bit integers most of the time
#if BLAS_INTEGER_KIND == 64
    integer, parameter :: blas_library_integer_kind = INT64
#else
    integer, parameter :: blas_library_integer_kind = INT32
#endif

! NWChem should be using 64-bit integers most of the time
#if NWCHEM_INTEGER_KIND == 32
    integer, parameter :: nwchem_integer_kind = INT32
#else
    integer, parameter :: nwchem_integer_kind = INT64
#endif

    interface
        ! BLAS Level 1
        subroutine dasum(n, x, incx, result)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx
            double precision, intent(in) :: x(*)
            double precision, intent(out) :: result
        end subroutine dasum

        subroutine daxpy(n, alpha, x, incx, y, incy)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx, incy
            double precision, intent(in) :: alpha, x(*)
            double precision, intent(inout) :: y(*)
        end subroutine daxpy

        subroutine dcopy(n, x, incx, y, incy)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx, incy
            double precision, intent(in) :: x(*)
            double precision, intent(out) :: y(*)
        end subroutine dcopy

        function ddot(n, x, incx, y, incy) result(result)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx, incy
            double precision, intent(in) :: x(*), y(*)
            double precision :: result
        end function ddot

        ! BLAS Level 2
        subroutine dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
            import blas_library_integer_kind
            character(len=1), intent(in) :: trans
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda, incx, incy
            double precision, intent(in) :: alpha, beta, a(lda,*), x(*)
            double precision, intent(inout) :: y(*)
        end subroutine dgemv

        subroutine dger(m, n, alpha, x, incx, y, incy, a, lda)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda, incx, incy
            double precision, intent(in) :: alpha, x(*), y(*)
            double precision, intent(inout) :: a(lda,*)
        end subroutine dger

        ! BLAS Level 3
        subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            import blas_library_integer_kind
            character(len=1), intent(in) :: transa, transb
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, k, lda, ldb, ldc
            double precision, intent(in) :: alpha, beta
            double precision, intent(in) :: a(lda,*), b(ldb,*)
            double precision, intent(out) :: c(ldc,*)
        end subroutine dgemm

        ! LAPACK routines
        subroutine dgebak(job, side, n, ilo, ihi, scale, m, v, ldv, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: job, side
            integer(kind=blas_library_integer_kind), intent(in) :: n, ilo, ihi, m, ldv
            double precision, intent(in) :: scale(*)
            double precision, intent(inout) :: v(ldv,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgebak

        subroutine dgebal(job, n, a, lda, ilo, ihi, scale, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: job
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda
            double precision, intent(inout) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(out) :: ilo, ihi
            double precision, intent(out) :: scale(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgebal

        ! Additional LAPACK routines
        subroutine dgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobvl, jobvr
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, ldvl, ldvr, lwork
            double precision, intent(inout) :: a(lda,*)
            double precision, intent(out) :: wr(*), wi(*), vl(ldvl,*), vr(ldvr,*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgeev

        subroutine dgeevx(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, &
                         ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, iwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: balanc, jobvl, jobvr, sense
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, ldvl, ldvr, lwork
            double precision, intent(inout) :: a(lda,*)
            double precision, intent(out) :: wr(*), wi(*), vl(ldvl,*), vr(ldvr,*), scale(*)
            double precision, intent(out) :: abnrm, rconde(*), rcondv(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: ilo, ihi, iwork(*), info
        end subroutine dgeevx

        subroutine dgehrd(n, ilo, ihi, a, lda, tau, work, lwork, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, ilo, ihi, lda, lwork
            double precision, intent(inout) :: a(lda,*)
            double precision, intent(out) :: tau(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgehrd

        subroutine dgels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: trans
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, nrhs, lda, ldb, lwork
            double precision, intent(inout) :: a(lda,*), b(ldb,*)
            double precision, intent(out) :: work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgels

        subroutine dgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, nrhs, lda, ldb, lwork
            double precision, intent(in) :: rcond
            double precision, intent(inout) :: a(lda,*), b(ldb,*)
            double precision, intent(out) :: s(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: rank, info
        end subroutine dgelss

        subroutine dgesv(n, nrhs, a, lda, ipiv, b, ldb, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, lda, ldb
            double precision, intent(inout) :: a(lda,*), b(ldb,*)
            integer(kind=blas_library_integer_kind), intent(out) :: ipiv(*), info
        end subroutine dgesv

        subroutine dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobu, jobvt
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda, ldu, ldvt, lwork
            double precision, intent(inout) :: a(lda,*)
            double precision, intent(out) :: s(*), u(ldu,*), vt(ldvt,*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgesvd

        subroutine dgetrf(m, n, a, lda, ipiv, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda
            double precision, intent(inout) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(out) :: ipiv(*), info
        end subroutine dgetrf

        subroutine dgetri(n, a, lda, ipiv, work, lwork, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, lwork
            double precision, intent(inout) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(in) :: ipiv(*)
            double precision, intent(out) :: work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgetri

        subroutine dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: trans
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, lda, ldb
            double precision, intent(in) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(in) :: ipiv(*)
            double precision, intent(inout) :: b(ldb,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgetrs

        ! Symmetric eigenvalue routines
        subroutine dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobz, uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, lwork
            double precision, intent(inout) :: a(lda,*)
            double precision, intent(out) :: w(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dsyev

        subroutine dsyevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, work, lwork, iwork, ifail, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobz, range, uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, il, iu, ldz, lwork
            double precision, intent(in) :: vl, vu, abstol
            double precision, intent(inout) :: a(lda,*)
            double precision, intent(out) :: w(*), z(ldz,*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: m, iwork(*), ifail(*), info
        end subroutine dsyevx

        ! Symmetric matrix factorization
        subroutine dsytrf(uplo, n, a, lda, ipiv, work, lwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, lwork
            double precision, intent(inout) :: a(lda,*)
            double precision, intent(out) :: work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: ipiv(*), info
        end subroutine dsytrf

        subroutine dsytri(uplo, n, a, lda, ipiv, work, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda
            double precision, intent(inout) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(in) :: ipiv(*)
            double precision, intent(out) :: work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dsytri

        ! Banded matrix routines
        subroutine dgbtrf(m, n, kl, ku, ab, ldab, ipiv, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, kl, ku, ldab
            double precision, intent(inout) :: ab(ldab,*)
            integer(kind=blas_library_integer_kind), intent(out) :: ipiv(*), info
        end subroutine dgbtrf

        subroutine dgbtrs(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: trans
            integer(kind=blas_library_integer_kind), intent(in) :: n, kl, ku, nrhs, ldab, ldb
            double precision, intent(in) :: ab(ldab,*)
            integer(kind=blas_library_integer_kind), intent(in) :: ipiv(*)
            double precision, intent(inout) :: b(ldb,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgbtrs

        ! Additional symmetric matrix routines
        subroutine dsytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, lda, ldb
            double precision, intent(in) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(in) :: ipiv(*)
            double precision, intent(inout) :: b(ldb,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dsytrs

        subroutine dsycon(uplo, n, a, lda, ipiv, anorm, rcond, work, iwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda
            double precision, intent(in) :: a(lda,*), anorm
            integer(kind=blas_library_integer_kind), intent(in) :: ipiv(*)
            double precision, intent(out) :: rcond, work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: iwork(*), info
        end subroutine dsycon

        subroutine dsyrfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, lda, ldaf, ldb, ldx
            double precision, intent(in) :: a(lda,*), af(ldaf,*), b(ldb,*)
            integer(kind=blas_library_integer_kind), intent(in) :: ipiv(*)
            double precision, intent(inout) :: x(ldx,*)
            double precision, intent(out) :: ferr(*), berr(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: iwork(*), info
        end subroutine dsyrfs

        ! Tridiagonal matrix routines
        subroutine dstev(jobz, n, d, e, z, ldz, work, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobz
            integer(kind=blas_library_integer_kind), intent(in) :: n, ldz
            double precision, intent(inout) :: d(*), e(*)
            double precision, intent(out) :: z(ldz,*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dstev

        subroutine dstevr(jobz, range, n, d, e, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobz, range
            integer(kind=blas_library_integer_kind), intent(in) :: n, il, iu, ldz, lwork, liwork
            double precision, intent(in) :: vl, vu, abstol
            double precision, intent(inout) :: d(*), e(*)
            double precision, intent(out) :: w(*), z(ldz,*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: m, isuppz(*), iwork(*), info
        end subroutine dstevr

        subroutine dpttrf(n, d, e, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n
            double precision, intent(inout) :: d(*), e(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dpttrf

        subroutine dpttrs(n, nrhs, d, e, b, ldb, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, ldb
            double precision, intent(in) :: d(*), e(*)
            double precision, intent(inout) :: b(ldb,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dpttrs

        ! Additional BLAS Level 1 routines
        subroutine dnrm2(n, x, incx, result)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx
            double precision, intent(in) :: x(*)
            double precision, intent(out) :: result
        end subroutine dnrm2

        subroutine drot(n, x, incx, y, incy, c, s)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx, incy
            double precision, intent(in) :: c, s
            double precision, intent(inout) :: x(*), y(*)
        end subroutine drot

        subroutine dscal(n, alpha, x, incx)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx
            double precision, intent(in) :: alpha
            double precision, intent(inout) :: x(*)
        end subroutine dscal

        subroutine dswap(n, x, incx, y, incy)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx, incy
            double precision, intent(inout) :: x(*), y(*)
        end subroutine dswap

        ! Additional BLAS Level 2 routines
        subroutine dsymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, incx, incy
            double precision, intent(in) :: alpha, beta, a(lda,*), x(*)
            double precision, intent(inout) :: y(*)
        end subroutine dsymv

        ! Additional BLAS Level 3 routines
        subroutine dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
            import blas_library_integer_kind
            character(len=1), intent(in) :: side, uplo
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda, ldb, ldc
            double precision, intent(in) :: alpha, beta, a(lda,*), b(ldb,*)
            double precision, intent(inout) :: c(ldc,*)
        end subroutine dsymm

        subroutine dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo, trans
            integer(kind=blas_library_integer_kind), intent(in) :: n, k, lda, ldc
            double precision, intent(in) :: alpha, beta, a(lda,*)
            double precision, intent(inout) :: c(ldc,*)
        end subroutine dsyrk

        subroutine dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo, trans
            integer(kind=blas_library_integer_kind), intent(in) :: n, k, lda, ldb, ldc
            double precision, intent(in) :: alpha, beta, a(lda,*), b(ldb,*)
            double precision, intent(inout) :: c(ldc,*)
        end subroutine dsyr2k

        ! Additional LAPACK routines
        subroutine dgtsv(n, nrhs, dl, d, du, b, ldb, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, ldb
            double precision, intent(inout) :: dl(*), d(*), du(*), b(ldb,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dgtsv

        subroutine dhseqr(job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz, work, lwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: job, compz
            integer(kind=blas_library_integer_kind), intent(in) :: n, ilo, ihi, ldh, ldz, lwork
            double precision, intent(inout) :: h(ldh,*), z(ldz,*)
            double precision, intent(out) :: wr(*), wi(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dhseqr

        subroutine dlacpy(uplo, m, n, a, lda, b, ldb)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda, ldb
            double precision, intent(in) :: a(lda,*)
            double precision, intent(out) :: b(ldb,*)
        end subroutine dlacpy

        subroutine dlagtf(n, a, lambda, b, c, tol, d, in, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n
            double precision, intent(in) :: lambda, tol
            double precision, intent(inout) :: a(*), b(*), c(*)
            double precision, intent(out) :: d(*)
            integer(kind=blas_library_integer_kind), intent(out) :: in(*), info
        end subroutine dlagtf

        subroutine dlagts(job, n, a, b, c, d, in, y, tol, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n
            character(len=1), intent(in) :: job
            double precision, intent(in) :: a(*), b(*), c(*), d(*), tol
            integer(kind=blas_library_integer_kind), intent(in) :: in(*)
            double precision, intent(inout) :: y(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dlagts

        function dlange(norm, m, n, a, lda, work) result(result)
            import blas_library_integer_kind
            character(len=1), intent(in) :: norm
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda
            double precision, intent(in) :: a(lda,*)
            double precision, intent(out) :: work(*)
            double precision :: result
        end function dlange

        subroutine dlarnv(idist, iseed, n, x)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: idist, n
            integer(kind=blas_library_integer_kind), intent(inout) :: iseed(*)
            double precision, intent(out) :: x(*)
        end subroutine dlarnv

        subroutine dlascl(type, kl, ku, cfrom, cto, m, n, a, lda, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: type
            integer(kind=blas_library_integer_kind), intent(in) :: kl, ku, m, n, lda
            double precision, intent(in) :: cfrom, cto
            double precision, intent(inout) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dlascl

        subroutine dlaset(uplo, m, n, alpha, beta, a, lda)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda
            double precision, intent(in) :: alpha, beta
            double precision, intent(out) :: a(lda,*)
        end subroutine dlaset

        ! Additional LAPACK routines
        subroutine dorghr(n, ilo, ihi, a, lda, tau, work, lwork, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, ilo, ihi, lda, lwork
            double precision, intent(inout) :: a(lda,*)
            double precision, intent(in) :: tau(*)
            double precision, intent(out) :: work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dorghr

        subroutine dpftrf(transr, uplo, n, a, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: transr, uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n
            double precision, intent(inout) :: a(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dpftrf

        subroutine dpftri(transr, uplo, n, a, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: transr, uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n
            double precision, intent(inout) :: a(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dpftri

        subroutine dposv(uplo, n, nrhs, a, lda, b, ldb, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, lda, ldb
            double precision, intent(inout) :: a(lda,*), b(ldb,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dposv

        subroutine dpotrf(uplo, n, a, lda, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda
            double precision, intent(inout) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dpotrf

        subroutine dpotri(uplo, n, a, lda, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda
            double precision, intent(inout) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dpotri

        subroutine dsfrk(transr, uplo, trans, n, k, alpha, a, lda, beta, c)
            import blas_library_integer_kind
            character(len=1), intent(in) :: transr, uplo, trans
            integer(kind=blas_library_integer_kind), intent(in) :: n, k, lda
            double precision, intent(in) :: alpha, beta, a(lda,*)
            double precision, intent(inout) :: c(*)
        end subroutine dsfrk

        subroutine dspsvx(fact, uplo, n, nrhs, a, afac, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: fact, uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, ldb, ldx
            double precision, intent(in) :: a(*), b(ldb,*)
            double precision, intent(inout) :: afac(*)
            integer(kind=blas_library_integer_kind), intent(inout) :: ipiv(*)
            double precision, intent(out) :: x(ldx,*), rcond, ferr(*), berr(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: iwork(*), info
        end subroutine dspsvx

        subroutine dstebz(range, order, n, vl, vu, il, iu, abstol, d, e, m, nsplit, w, iblock, isplit, work, iwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: range, order
            integer(kind=blas_library_integer_kind), intent(in) :: n, il, iu
            double precision, intent(in) :: vl, vu, abstol, d(*), e(*)
            double precision, intent(out) :: w(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: m, nsplit, iblock(*), isplit(*), iwork(*), info
        end subroutine dstebz

        subroutine dstein(n, d, e, m, w, iblock, isplit, z, ldz, work, iwork, ifail, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, m, ldz
            double precision, intent(in) :: d(*), e(*), w(*)
            integer(kind=blas_library_integer_kind), intent(in) :: iblock(*), isplit(*)
            double precision, intent(out) :: z(ldz,*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: iwork(*), ifail(*), info
        end subroutine dstein

        subroutine dsterf(n, d, e, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n
            double precision, intent(inout) :: d(*), e(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dsterf

        subroutine dsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobz, uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, lwork, liwork
            double precision, intent(inout) :: a(lda,*)
            double precision, intent(out) :: w(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: iwork(*), info
        end subroutine dsyevd

        subroutine dsygv(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: itype, n, lda, ldb, lwork
            character(len=1), intent(in) :: jobz, uplo
            double precision, intent(inout) :: a(lda,*), b(ldb,*)
            double precision, intent(out) :: w(*), work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dsygv

        subroutine dsysv(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, lwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, lda, ldb, lwork
            double precision, intent(inout) :: a(lda,*), b(ldb,*)
            double precision, intent(out) :: work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: ipiv(*), info
        end subroutine dsysv

        ! Additional LAPACK routines
        subroutine dtrtri(uplo, diag, n, a, lda, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo, diag
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda
            double precision, intent(inout) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine dtrtri

        subroutine dtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
            import blas_library_integer_kind
            character(len=1), intent(in) :: side, uplo, transa, diag
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda, ldb
            double precision, intent(in) :: alpha, a(lda,*)
            double precision, intent(inout) :: b(ldb,*)
        end subroutine dtrmm

        subroutine dtfsm(transr, side, uplo, trans, unit, m, n, alpha, a, b, ldb)
            import blas_library_integer_kind
            character(len=1), intent(in) :: transr, side, uplo, trans, unit
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, ldb
            double precision, intent(in) :: alpha, a(*)
            double precision, intent(inout) :: b(ldb,*)
        end subroutine dtfsm

        subroutine dtrevc(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: side, howmny
            integer(kind=blas_library_integer_kind), intent(in) :: n, ldt, ldvl, ldvr, mm
            logical, intent(in) :: select(*)
            double precision, intent(in) :: t(ldt,*)
            double precision, intent(inout) :: vl(ldvl,*), vr(ldvr,*)
            double precision, intent(out) :: work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: m, info
        end subroutine dtrevc

        ! Complex BLAS Level 1
        subroutine zaxpy(n, alpha, x, incx, y, incy)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx, incy
            complex(kind=8), intent(in) :: alpha, x(*)
            complex(kind=8), intent(inout) :: y(*)
        end subroutine zaxpy

        subroutine zcopy(n, x, incx, y, incy)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx, incy
            complex(kind=8), intent(in) :: x(*)
            complex(kind=8), intent(out) :: y(*)
        end subroutine zcopy

        subroutine zdotc(n, x, incx, y, incy, result)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx, incy
            complex(kind=8), intent(in) :: x(*), y(*)
            complex(kind=8), intent(out) :: result
        end subroutine zdotc

        subroutine zscal(n, alpha, x, incx)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: n, incx
            complex(kind=8), intent(in) :: alpha
            complex(kind=8), intent(inout) :: x(*)
        end subroutine zscal

        ! Complex BLAS Level 3
        subroutine zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            import blas_library_integer_kind
            character(len=1), intent(in) :: transa, transb
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, k, lda, ldb, ldc
            complex(kind=8), intent(in) :: alpha, beta
            complex(kind=8), intent(in) :: a(lda,*), b(ldb,*)
            complex(kind=8), intent(inout) :: c(ldc,*)
        end subroutine zgemm

        subroutine zsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo, trans
            integer(kind=blas_library_integer_kind), intent(in) :: n, k, lda, ldc
            complex(kind=8), intent(in) :: alpha, beta
            complex(kind=8), intent(in) :: a(lda,*)
            complex(kind=8), intent(inout) :: c(ldc,*)
        end subroutine zsyrk

        ! Complex LAPACK routines
        subroutine zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobvl, jobvr
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, ldvl, ldvr, lwork
            complex(kind=8), intent(inout) :: a(lda,*)
            complex(kind=8), intent(out) :: w(*), vl(ldvl,*), vr(ldvr,*), work(*)
            double precision, intent(out) :: rwork(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine zgeev

        subroutine zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobu, jobvt
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda, ldu, ldvt, lwork
            complex(kind=8), intent(inout) :: a(lda,*)
            double precision, intent(out) :: s(*)
            complex(kind=8), intent(out) :: u(ldu,*), vt(ldvt,*), work(*)
            double precision, intent(out) :: rwork(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine zgesvd

        subroutine zgetrf(m, n, a, lda, ipiv, info)
            import blas_library_integer_kind
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda
            complex(kind=8), intent(inout) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(out) :: ipiv(*), info
        end subroutine zgetrf

        subroutine zgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: trans
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, lda, ldb
            complex(kind=8), intent(in) :: a(lda,*)
            integer(kind=blas_library_integer_kind), intent(in) :: ipiv(*)
            complex(kind=8), intent(inout) :: b(ldb,*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine zgetrs

        subroutine zheev(jobz, uplo, n, a, lda, w, work, lwork, rwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: jobz, uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, lwork
            complex(kind=8), intent(inout) :: a(lda,*)
            double precision, intent(out) :: w(*)
            complex(kind=8), intent(out) :: work(*)
            double precision, intent(out) :: rwork(*)
            integer(kind=blas_library_integer_kind), intent(out) :: info
        end subroutine zheev

        subroutine zlacp2(uplo, m, n, a, lda, b, ldb)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda, ldb
            double precision, intent(in) :: a(lda,*)
            complex(kind=8), intent(out) :: b(ldb,*)
        end subroutine zlacp2

        subroutine zlacpy(uplo, m, n, a, lda, b, ldb)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, lda, ldb
            complex(kind=8), intent(in) :: a(lda,*)
            complex(kind=8), intent(out) :: b(ldb,*)
        end subroutine zlacpy

        subroutine zsysv(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, lwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, nrhs, lda, ldb, lwork
            complex(kind=8), intent(inout) :: a(lda,*), b(ldb,*)
            complex(kind=8), intent(out) :: work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: ipiv(*), info
        end subroutine zsysv

        subroutine zsytrf(uplo, n, a, lda, ipiv, work, lwork, info)
            import blas_library_integer_kind
            character(len=1), intent(in) :: uplo
            integer(kind=blas_library_integer_kind), intent(in) :: n, lda, lwork
            complex(kind=8), intent(inout) :: a(lda,*)
            complex(kind=8), intent(out) :: work(*)
            integer(kind=blas_library_integer_kind), intent(out) :: ipiv(*), info
        end subroutine zsytrf
    end interface

    contains

    ! BLAS Level 1 wrappers
    subroutine yasum(n, x, incx, result)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx
        double precision, intent(in) :: x(*)
        double precision, intent(out) :: result
        integer(kind=blas_library_integer_kind) :: n_int, incx_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        call dasum(n_int, x, incx_int, result)
    end subroutine yasum

    subroutine yaxpy(n, alpha, x, incx, y, incy)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx, incy
        double precision, intent(in) :: alpha, x(*)
        double precision, intent(inout) :: y(*)
        integer(kind=blas_library_integer_kind) :: n_int, incx_int, incy_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call daxpy(n_int, alpha, x, incx_int, y, incy_int)
    end subroutine yaxpy

    subroutine ycopy(n, x, incx, y, incy)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx, incy
        double precision, intent(in) :: x(*)
        double precision, intent(out) :: y(*)
        integer(kind=blas_library_integer_kind) :: n_int, incx_int, incy_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call dcopy(n_int, x, incx_int, y, incy_int)
    end subroutine ycopy

    function ydot(n, x, incx, y, incy) result(result)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx, incy
        double precision, intent(in) :: x(*), y(*)
        double precision :: result
        integer(kind=blas_library_integer_kind) :: n_int, incx_int, incy_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        result = ddot(n_int, x, incx_int, y, incy_int)
    end function ydot

    ! BLAS Level 2 wrappers
    subroutine ygemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
        character(len=1), intent(in) :: trans
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda, incx, incy
        double precision, intent(in) :: alpha, beta, a(lda,*), x(*)
        double precision, intent(inout) :: y(*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, incx_int, incy_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call dgemv(trans, m_int, n_int, alpha, a, lda_int, x, incx_int, beta, y, incy_int)
    end subroutine ygemv

    subroutine yger(m, n, alpha, x, incx, y, incy, a, lda)
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda, incx, incy
        double precision, intent(in) :: alpha, x(*), y(*)
        double precision, intent(inout) :: a(lda,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, incx_int, incy_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call dger(m_int, n_int, alpha, x, incx_int, y, incy_int, a, lda_int)
    end subroutine yger

    ! BLAS Level 3 wrapper
    subroutine ygemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        character(len=1), intent(in) :: transa, transb
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, k, lda, ldb, ldc
        double precision, intent(in) :: alpha, beta
        double precision, intent(in) :: a(lda,*), b(ldb,*)
        double precision, intent(out) :: c(ldc,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, k_int, lda_int, ldb_int, ldc_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        k_int = int(k, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        ldc_int = int(ldc, kind=blas_library_integer_kind)
        call dgemm(transa, transb, m_int, n_int, k_int, alpha, a, lda_int, b, ldb_int, beta, c, ldc_int)
    end subroutine ygemm

    ! LAPACK wrappers
    subroutine ygebak(job, side, n, ilo, ihi, scale, m, v, ldv, info)
        character(len=1), intent(in) :: job, side
        integer(kind=nwchem_integer_kind), intent(in) :: n, ilo, ihi, m, ldv
        double precision, intent(in) :: scale(*)
        double precision, intent(inout) :: v(ldv,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, ilo_int, ihi_int, m_int, ldv_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        ilo_int = int(ilo, kind=blas_library_integer_kind)
        ihi_int = int(ihi, kind=blas_library_integer_kind)
        m_int = int(m, kind=blas_library_integer_kind)
        ldv_int = int(ldv, kind=blas_library_integer_kind)
        call dgebak(job, side, n_int, ilo_int, ihi_int, scale, m_int, v, ldv_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygebak

    subroutine ygebal(job, n, a, lda, ilo, ihi, scale, info)
        character(len=1), intent(in) :: job
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda
        double precision, intent(inout) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(out) :: ilo, ihi
        double precision, intent(out) :: scale(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, ilo_int, ihi_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        call dgebal(job, n_int, a, lda_int, ilo_int, ihi_int, scale, info_int)
        ilo = int(ilo_int, kind=nwchem_integer_kind)
        ihi = int(ihi_int, kind=nwchem_integer_kind)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygebal

    ! Additional LAPACK wrappers
    subroutine ygeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info)
        character(len=1), intent(in) :: jobvl, jobvr
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda, ldvl, ldvr, lwork
        double precision, intent(inout) :: a(lda,*)
        double precision, intent(out) :: wr(*), wi(*), vl(ldvl,*), vr(ldvr,*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, ldvl_int, ldvr_int, lwork_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldvl_int = int(ldvl, kind=blas_library_integer_kind)
        ldvr_int = int(ldvr, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call dgeev(jobvl, jobvr, n_int, a, lda_int, wr, wi, vl, ldvl_int, vr, ldvr_int, work, lwork_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygeev

    subroutine ygeevx(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, &
                     ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, iwork, info)
        character(len=1), intent(in) :: balanc, jobvl, jobvr, sense
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda, ldvl, ldvr, lwork
        double precision, intent(inout) :: a(lda,*)
        double precision, intent(out) :: wr(*), wi(*), vl(ldvl,*), vr(ldvr,*), scale(*)
        double precision, intent(out) :: abnrm, rconde(*), rcondv(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: ilo, ihi, iwork(*), info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, ldvl_int, ldvr_int, lwork_int
        integer(kind=blas_library_integer_kind) :: ilo_int, ihi_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: iwork_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldvl_int = int(ldvl, kind=blas_library_integer_kind)
        ldvr_int = int(ldvr, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        allocate( iwork_int(2*n-2) )
        call dgeevx(balanc, jobvl, jobvr, sense, n_int, a, lda_int, wr, wi, vl, ldvl_int, vr, ldvr_int, &
                    ilo_int, ihi_int, scale, abnrm, rconde, rcondv, work, lwork_int, iwork_int, info_int)
        ilo = int(ilo_int, kind=nwchem_integer_kind)
        ihi = int(ihi_int, kind=nwchem_integer_kind)
        iwork(1:2*n-2) = int(iwork_int(1:2*n-2), kind=nwchem_integer_kind)
        deallocate( iwork_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygeevx

    subroutine ygehrd(n, ilo, ihi, a, lda, tau, work, lwork, info)
        integer(kind=nwchem_integer_kind), intent(in) :: n, ilo, ihi, lda, lwork
        double precision, intent(inout) :: a(lda,*)
        double precision, intent(out) :: tau(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, ilo_int, ihi_int, lda_int, lwork_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        ilo_int = int(ilo, kind=blas_library_integer_kind)
        ihi_int = int(ihi, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call dgehrd(n_int, ilo_int, ihi_int, a, lda_int, tau, work, lwork_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygehrd

    subroutine ygels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info)
        character(len=1), intent(in) :: trans
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, nrhs, lda, ldb, lwork
        double precision, intent(inout) :: a(lda,*), b(ldb,*)
        double precision, intent(out) :: work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: m_int, n_int, nrhs_int, lda_int, ldb_int, lwork_int, info_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call dgels(trans, m_int, n_int, nrhs_int, a, lda_int, b, ldb_int, work, lwork_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygels

    subroutine ygelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, info)
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, nrhs, lda, ldb, lwork
        double precision, intent(in) :: rcond
        double precision, intent(inout) :: a(lda,*), b(ldb,*)
        double precision, intent(out) :: s(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: rank, info
        integer(kind=blas_library_integer_kind) :: m_int, n_int, nrhs_int, lda_int, ldb_int, lwork_int
        integer(kind=blas_library_integer_kind) :: rank_int, info_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call dgelss(m_int, n_int, nrhs_int, a, lda_int, b, ldb_int, s, rcond, rank_int, work, lwork_int, info_int)
        rank = int(rank_int, kind=nwchem_integer_kind)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygelss

    ! subroutine ygesv(n, nrhs, a, lda, ipiv, b, ldb, info)
    !     integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, lda, ldb
    !     double precision, intent(inout) :: a(lda,*), b(ldb,*)
    !     integer(kind=nwchem_integer_kind), intent(out) :: ipiv(*), info
    !     integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, lda_int, ldb_int, info_int
    !     n_int = int(n, kind=blas_library_integer_kind)
    !     nrhs_int = int(nrhs, kind=blas_library_integer_kind)
    !     lda_int = int(lda, kind=blas_library_integer_kind)
    !     ldb_int = int(ldb, kind=blas_library_integer_kind)
    !     call dgesv(n_int, nrhs_int, a, lda_int, ipiv, b, ldb_int, info_int)
    !     info = int(info_int, kind=nwchem_integer_kind)
    ! end subroutine ygesv

    subroutine ygesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
        character(len=1), intent(in) :: jobu, jobvt
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda, ldu, ldvt, lwork
        double precision, intent(inout) :: a(lda,*)
        double precision, intent(out) :: s(*), u(ldu,*), vt(ldvt,*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, ldu_int, ldvt_int, lwork_int, info_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldu_int = int(ldu, kind=blas_library_integer_kind)
        ldvt_int = int(ldvt, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call dgesvd(jobu, jobvt, m_int, n_int, a, lda_int, s, u, ldu_int, vt, ldvt_int, work, lwork_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygesvd

    ! subroutine ygetrf(m, n, a, lda, ipiv, info)
    !     integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda
    !     double precision, intent(inout) :: a(lda,*)
    !     integer(kind=nwchem_integer_kind), intent(out) :: ipiv(*), info
    !     integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, info_int
    !     integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
    !     m_int = int(m, kind=blas_library_integer_kind)
    !     n_int = int(n, kind=blas_library_integer_kind)
    !     lda_int = int(lda, kind=blas_library_integer_kind)
    !     allocate( ipiv_int(min(m,n)) )
    !     call dgetrf(m_int, n_int, a, lda_int, ipiv_int, info_int)
    !     ipiv = int(ipiv_int(1:min(m,n)), kind=nwchem_integer_kind)
    !     deallocate( ipiv_int )
    !     info = int(info_int, kind=nwchem_integer_kind)
    ! end subroutine ygetrf

    subroutine ygetri(n, a, lda, ipiv, work, lwork, info)
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda, lwork
        double precision, intent(inout) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(in) :: ipiv(*)
        double precision, intent(out) :: work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, lwork_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        allocate( ipiv_int(n) )
        ipiv_int = int(ipiv(1:n), kind=blas_library_integer_kind)
        call dgetri(n_int, a, lda_int, ipiv_int, work, lwork_int, info_int)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygetri

    subroutine ygetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
        character(len=1), intent(in) :: trans
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, lda, ldb
        double precision, intent(in) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(in) :: ipiv(*)
        double precision, intent(inout) :: b(ldb,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, lda_int, ldb_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        allocate( ipiv_int(n) )
        ipiv_int = int(ipiv(1:n), kind=blas_library_integer_kind)
        call dgetrs(trans, n_int, nrhs_int, a, lda_int, ipiv_int, b, ldb_int, info_int)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygetrs

    ! Symmetric eigenvalue wrappers
    subroutine ysyev(jobz, uplo, n, a, lda, w, work, lwork, info)
        character(len=1), intent(in) :: jobz, uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda, lwork
        double precision, intent(inout) :: a(lda,*)
        double precision, intent(out) :: w(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, lwork_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call dsyev(jobz, uplo, n_int, a, lda_int, w, work, lwork_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ysyev

    ! subroutine ysyevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, work, lwork, iwork, ifail, info)
    !     character(len=1), intent(in) :: jobz, range, uplo
    !     integer(kind=nwchem_integer_kind), intent(in) :: n, lda, il, iu, ldz, lwork
    !     double precision, intent(in) :: vl, vu, abstol
    !     double precision, intent(inout) :: a(lda,*)
    !     double precision, intent(out) :: w(*), z(ldz,*), work(*)
    !     integer(kind=nwchem_integer_kind), intent(out) :: m, iwork(*), ifail(*), info
    !     integer(kind=blas_library_integer_kind) :: n_int, lda_int, il_int, iu_int, ldz_int, lwork_int
    !     integer(kind=blas_library_integer_kind) :: m_int, info_int
    !     n_int = int(n, kind=blas_library_integer_kind)
    !     lda_int = int(lda, kind=blas_library_integer_kind)
    !     il_int = int(il, kind=blas_library_integer_kind)
    !     iu_int = int(iu, kind=blas_library_integer_kind)
    !     ldz_int = int(ldz, kind=blas_library_integer_kind)
    !     lwork_int = int(lwork, kind=blas_library_integer_kind)
    !     call dsyevx(jobz, range, uplo, n_int, a, lda_int, vl, vu, il_int, iu_int, abstol, m_int, w, z, ldz_int, &
    !                work, lwork_int, iwork, ifail, info_int)
    !     m = int(m_int, kind=nwchem_integer_kind)
    !     info = int(info_int, kind=nwchem_integer_kind)
    ! end subroutine ysyevx

    ! Symmetric matrix factorization
    ! subroutine ysytrf(uplo, n, a, lda, ipiv, work, lwork, info)
    !     character(len=1), intent(in) :: uplo
    !     integer(kind=nwchem_integer_kind), intent(in) :: n, lda, lwork
    !     double precision, intent(inout) :: a(lda,*)
    !     integer(kind=nwchem_integer_kind), intent(out) :: ipiv(*), info
    !     double precision, intent(out) :: work(*)
    !     integer(kind=blas_library_integer_kind) :: n_int, lda_int, lwork_int, info_int
    !     integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
    !     n_int = int(n, kind=blas_library_integer_kind)
    !     lda_int = int(lda, kind=blas_library_integer_kind)
    !     lwork_int = int(lwork, kind=blas_library_integer_kind)
    !     allocate( ipiv_int(n) )
    !     call dsytrf(uplo, n_int, a, lda_int, ipiv_int, work, lwork_int, info_int)
    !     ipiv = int(ipiv_int(1:n), kind=nwchem_integer_kind)
    !     deallocate( ipiv_int )
    !     info = int(info_int, kind=nwchem_integer_kind)
    ! end subroutine ysytrf

    subroutine ysytri(uplo, n, a, lda, ipiv, work, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda
        double precision, intent(inout) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(in) :: ipiv(*)
        double precision, intent(out) :: work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        allocate( ipiv_int(n) )
        ipiv_int = int(ipiv(1:n), kind=blas_library_integer_kind)
        call dsytri(uplo, n_int, a, lda_int, ipiv_int, work, info_int)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ysytri

    ! Banded matrix wrappers
    ! subroutine ygbtrf(m, n, kl, ku, ab, ldab, ipiv, info)
    !     integer(kind=nwchem_integer_kind), intent(in) :: m, n, kl, ku, ldab
    !     double precision, intent(inout) :: ab(ldab,*)
    !     integer(kind=nwchem_integer_kind), intent(out) :: ipiv(*), info
    !     integer(kind=blas_library_integer_kind) :: m_int, n_int, kl_int, ku_int, ldab_int, info_int
    !     integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
    !     m_int = int(m, kind=blas_library_integer_kind)
    !     n_int = int(n, kind=blas_library_integer_kind)
    !     kl_int = int(kl, kind=blas_library_integer_kind)
    !     ku_int = int(ku, kind=blas_library_integer_kind)
    !     ldab_int = int(ldab, kind=blas_library_integer_kind)
    !     allocate( ipiv_int(min(m,n)) )
    !     call dgbtrf(m_int, n_int, kl_int, ku_int, ab, ldab_int, ipiv_int, info_int)
    !     ipiv = int(ipiv_int(1:min(m,n)), kind=nwchem_integer_kind)
    !     deallocate( ipiv_int )
    !     info = int(info_int, kind=nwchem_integer_kind)
    ! end subroutine ygbtrf

    subroutine ygbtrs(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info)
        character(len=1), intent(in) :: trans
        integer(kind=nwchem_integer_kind), intent(in) :: n, kl, ku, nrhs, ldab, ldb
        double precision, intent(in) :: ab(ldab,*)
        integer(kind=nwchem_integer_kind), intent(in) :: ipiv(*)
        double precision, intent(inout) :: b(ldb,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, kl_int, ku_int, nrhs_int, ldab_int, ldb_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        kl_int = int(kl, kind=blas_library_integer_kind)
        ku_int = int(ku, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        ldab_int = int(ldab, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        allocate( ipiv_int(n) )
        ipiv_int = int(ipiv(1:n), kind=blas_library_integer_kind)
        call dgbtrs(trans, n_int, kl_int, ku_int, nrhs_int, ab, ldab_int, ipiv_int, b, ldb_int, info_int)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygbtrs

    ! Additional symmetric matrix wrappers
    subroutine ysytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, lda, ldb
        double precision, intent(in) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(in) :: ipiv(*)
        double precision, intent(inout) :: b(ldb,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, lda_int, ldb_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        allocate( ipiv_int(n) )
        ipiv_int = int(ipiv(1:n), kind=blas_library_integer_kind)
        call dsytrs(uplo, n_int, nrhs_int, a, lda_int, ipiv_int, b, ldb_int, info_int)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ysytrs

    subroutine ysycon(uplo, n, a, lda, ipiv, anorm, rcond, work, iwork, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda
        double precision, intent(in) :: a(lda,*), anorm
        integer(kind=nwchem_integer_kind), intent(in) :: ipiv(*)
        double precision, intent(out) :: rcond, work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: iwork(*), info ! iwork is unused
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:), iwork_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        allocate( ipiv_int(n), iwork_int(n) )
        ipiv_int = int(ipiv(1:n), kind=blas_library_integer_kind)
        call dsycon(uplo, n_int, a, lda_int, ipiv_int, anorm, rcond, work, iwork_int, info_int)
        deallocate( ipiv_int, iwork_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ysycon

    subroutine ysyrfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, lda, ldaf, ldb, ldx
        double precision, intent(in) :: a(lda,*), af(ldaf,*), b(ldb,*)
        integer(kind=nwchem_integer_kind), intent(in) :: ipiv(*)
        double precision, intent(inout) :: x(ldx,*)
        double precision, intent(out) :: ferr(*), berr(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: iwork(*), info ! iwork is unused
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, lda_int, ldaf_int, ldb_int, ldx_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:), iwork_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldaf_int = int(ldaf, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        ldx_int = int(ldx, kind=blas_library_integer_kind)
        allocate( ipiv_int(n), iwork_int(n) )
        ipiv_int = int(ipiv(1:n), kind=blas_library_integer_kind)
        call dsyrfs(uplo, n_int, nrhs_int, a, lda_int, af, ldaf_int, ipiv_int, b, ldb_int, x, ldx_int, &
                    ferr, berr, work, iwork_int, info_int)
        deallocate( ipiv_int, iwork_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ysyrfs

    ! Tridiagonal matrix wrappers
    subroutine ystev(jobz, n, d, e, z, ldz, work, info)
        character(len=1), intent(in) :: jobz
        integer(kind=nwchem_integer_kind), intent(in) :: n, ldz
        double precision, intent(inout) :: d(*), e(*)
        double precision, intent(out) :: z(ldz,*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, ldz_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        ldz_int = int(ldz, kind=blas_library_integer_kind)
        call dstev(jobz, n_int, d, e, z, ldz_int, work, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ystev

    ! subroutine ystevr(jobz, range, n, d, e, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info)
    !     character(len=1), intent(in) :: jobz, range
    !     integer(kind=nwchem_integer_kind), intent(in) :: n, il, iu, ldz, lwork, liwork
    !     double precision, intent(in) :: vl, vu, abstol
    !     double precision, intent(inout) :: d(*), e(*)
    !     double precision, intent(out) :: w(*), z(ldz,*), work(*)
    !     integer(kind=nwchem_integer_kind), intent(out) :: m, isuppz(*), iwork(*), info
    !     integer(kind=blas_library_integer_kind) :: n_int, il_int, iu_int, ldz_int, lwork_int, liwork_int
    !     integer(kind=blas_library_integer_kind) :: m_int, info_int
    !     n_int = int(n, kind=blas_library_integer_kind)
    !     il_int = int(il, kind=blas_library_integer_kind)
    !     iu_int = int(iu, kind=blas_library_integer_kind)
    !     ldz_int = int(ldz, kind=blas_library_integer_kind)
    !     lwork_int = int(lwork, kind=blas_library_integer_kind)
    !     liwork_int = int(liwork, kind=blas_library_integer_kind)
    !     call dstevr(jobz, range, n_int, d, e, vl, vu, il_int, iu_int, abstol, m_int, w, z, ldz_int, &
    !                isuppz, work, lwork_int, iwork, liwork_int, info_int)
    !     m = int(m_int, kind=nwchem_integer_kind)
    !     info = int(info_int, kind=nwchem_integer_kind)
    ! end subroutine ystevr

    subroutine ypttrf(n, d, e, info)
        integer(kind=nwchem_integer_kind), intent(in) :: n
        double precision, intent(inout) :: d(*), e(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        call dpttrf(n_int, d, e, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ypttrf

    subroutine ypttrs(n, nrhs, d, e, b, ldb, info)
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, ldb
        double precision, intent(in) :: d(*), e(*)
        double precision, intent(inout) :: b(ldb,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, ldb_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        call dpttrs(n_int, nrhs_int, d, e, b, ldb_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ypttrs

    ! Additional BLAS Level 1 wrappers
    subroutine ynrm2(n, x, incx, result)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx
        double precision, intent(in) :: x(*)
        double precision, intent(out) :: result
        integer(kind=blas_library_integer_kind) :: n_int, incx_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        call dnrm2(n_int, x, incx_int, result)
    end subroutine ynrm2

    subroutine yrot(n, x, incx, y, incy, c, s)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx, incy
        double precision, intent(in) :: c, s
        double precision, intent(inout) :: x(*), y(*)
        integer(kind=blas_library_integer_kind) :: n_int, incx_int, incy_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call drot(n_int, x, incx_int, y, incy_int, c, s)
    end subroutine yrot

    subroutine yscal(n, alpha, x, incx)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx
        double precision, intent(in) :: alpha
        double precision, intent(inout) :: x(*)
        integer(kind=blas_library_integer_kind) :: n_int, incx_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        call dscal(n_int, alpha, x, incx_int)
    end subroutine yscal

    subroutine yswap(n, x, incx, y, incy)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx, incy
        double precision, intent(inout) :: x(*), y(*)
        integer(kind=blas_library_integer_kind) :: n_int, incx_int, incy_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call dswap(n_int, x, incx_int, y, incy_int)
    end subroutine yswap

    ! Additional BLAS Level 2 wrappers
    subroutine ysymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda, incx, incy
        double precision, intent(in) :: alpha, beta, a(lda,*), x(*)
        double precision, intent(inout) :: y(*)
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, incx_int, incy_int
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call dsymv(uplo, n_int, alpha, a, lda_int, x, incx_int, beta, y, incy_int)
    end subroutine ysymv

    ! Additional BLAS Level 3 wrappers
    subroutine ysymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
        character(len=1), intent(in) :: side, uplo
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda, ldb, ldc
        double precision, intent(in) :: alpha, beta, a(lda,*), b(ldb,*)
        double precision, intent(inout) :: c(ldc,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, ldb_int, ldc_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        ldc_int = int(ldc, kind=blas_library_integer_kind)
        call dsymm(side, uplo, m_int, n_int, alpha, a, lda_int, b, ldb_int, beta, c, ldc_int)
    end subroutine ysymm

    subroutine ysyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
        character(len=1), intent(in) :: uplo, trans
        integer(kind=nwchem_integer_kind), intent(in) :: n, k, lda, ldc
        double precision, intent(in) :: alpha, beta, a(lda,*)
        double precision, intent(inout) :: c(ldc,*)
        integer(kind=blas_library_integer_kind) :: n_int, k_int, lda_int, ldc_int
        n_int = int(n, kind=blas_library_integer_kind)
        k_int = int(k, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldc_int = int(ldc, kind=blas_library_integer_kind)
        call dsyrk(uplo, trans, n_int, k_int, alpha, a, lda_int, beta, c, ldc_int)
    end subroutine ysyrk

    subroutine ysyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        character(len=1), intent(in) :: uplo, trans
        integer(kind=nwchem_integer_kind), intent(in) :: n, k, lda, ldb, ldc
        double precision, intent(in) :: alpha, beta, a(lda,*), b(ldb,*)
        double precision, intent(inout) :: c(ldc,*)
        integer(kind=blas_library_integer_kind) :: n_int, k_int, lda_int, ldb_int, ldc_int
        n_int = int(n, kind=blas_library_integer_kind)
        k_int = int(k, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        ldc_int = int(ldc, kind=blas_library_integer_kind)
        call dsyr2k(uplo, trans, n_int, k_int, alpha, a, lda_int, b, ldb_int, beta, c, ldc_int)
    end subroutine ysyr2k

    ! Additional LAPACK wrappers
    subroutine ygtsv(n, nrhs, dl, d, du, b, ldb, info)
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, ldb
        double precision, intent(inout) :: dl(*), d(*), du(*), b(ldb,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, ldb_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        call dgtsv(n_int, nrhs_int, dl, d, du, b, ldb_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ygtsv

    subroutine yhseqr(job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz, work, lwork, info)
        character(len=1), intent(in) :: job, compz
        integer(kind=nwchem_integer_kind), intent(in) :: n, ilo, ihi, ldh, ldz, lwork
        double precision, intent(inout) :: h(ldh,*), z(ldz,*)
        double precision, intent(out) :: wr(*), wi(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, ilo_int, ihi_int, ldh_int, ldz_int, lwork_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        ilo_int = int(ilo, kind=blas_library_integer_kind)
        ihi_int = int(ihi, kind=blas_library_integer_kind)
        ldh_int = int(ldh, kind=blas_library_integer_kind)
        ldz_int = int(ldz, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call dhseqr(job, compz, n_int, ilo_int, ihi_int, h, ldh_int, wr, wi, z, ldz_int, work, lwork_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yhseqr

    subroutine ylacpy(uplo, m, n, a, lda, b, ldb)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda, ldb
        double precision, intent(in) :: a(lda,*)
        double precision, intent(out) :: b(ldb,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, ldb_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        call dlacpy(uplo, m_int, n_int, a, lda_int, b, ldb_int)
    end subroutine ylacpy

    ! subroutine ylagtf(n, a, lambda, b, c, tol, d, in, info)
    !     integer(kind=nwchem_integer_kind), intent(in) :: n
    !     double precision, intent(in) :: lambda, tol
    !     double precision, intent(inout) :: a(*), b(*), c(*)
    !     double precision, intent(out) :: d(*)
    !     integer(kind=nwchem_integer_kind), intent(out) :: in(*), info
    !     integer(kind=blas_library_integer_kind) :: n_int, info_int
    !     n_int = int(n, kind=blas_library_integer_kind)
    !     call dlagtf(n_int, a, lambda, b, c, tol, d, in, info_int)
    !     info = int(info_int, kind=nwchem_integer_kind)
    ! end subroutine ylagtf

    ! subroutine ylagts(job, n, a, b, c, d, in, y, tol, info)
    !     integer(kind=nwchem_integer_kind), intent(in) :: n
    !     character(len=1), intent(in) :: job
    !     double precision, intent(in) :: a(*), b(*), c(*), d(*), tol
    !     integer(kind=nwchem_integer_kind), intent(in) :: in(*)
    !     double precision, intent(inout) :: y(*)
    !     integer(kind=nwchem_integer_kind), intent(out) :: info
    !     integer(kind=blas_library_integer_kind) :: n_int, info_int
    !     n_int = int(n, kind=blas_library_integer_kind)
    !     call dlagts(job, n_int, a, b, c, d, in, y, tol, info_int)
    !     info = int(info_int, kind=nwchem_integer_kind)
    ! end subroutine ylagts

    function ylange(norm, m, n, a, lda, work) result(result)
        character(len=1), intent(in) :: norm
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda
        double precision, intent(in) :: a(lda,*)
        double precision, intent(out) :: work(*)
        double precision :: result
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        result = dlange(norm, m_int, n_int, a, lda_int, work)
    end function ylange

    ! subroutine ylarnv(idist, iseed, n, x)
    !     integer(kind=nwchem_integer_kind), intent(in) :: idist, n
    !     integer(kind=nwchem_integer_kind), intent(inout) :: iseed(*)
    !     double precision, intent(out) :: x(*)
    !     integer(kind=blas_library_integer_kind) :: idist_int, n_int
    !     idist_int = int(idist, kind=blas_library_integer_kind)
    !     n_int = int(n, kind=blas_library_integer_kind)
    !     call dlarnv(idist_int, iseed, n_int, x)
    ! end subroutine ylarnv

    subroutine ylascl(type, kl, ku, cfrom, cto, m, n, a, lda, info)
        character(len=1), intent(in) :: type
        integer(kind=nwchem_integer_kind), intent(in) :: kl, ku, m, n, lda
        double precision, intent(in) :: cfrom, cto
        double precision, intent(inout) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: kl_int, ku_int, m_int, n_int, lda_int, info_int
        kl_int = int(kl, kind=blas_library_integer_kind)
        ku_int = int(ku, kind=blas_library_integer_kind)
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        call dlascl(type, kl_int, ku_int, cfrom, cto, m_int, n_int, a, lda_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ylascl

    subroutine ylaset(uplo, m, n, alpha, beta, a, lda)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda
        double precision, intent(in) :: alpha, beta
        double precision, intent(out) :: a(lda,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        call dlaset(uplo, m_int, n_int, alpha, beta, a, lda_int)
    end subroutine ylaset

    ! Additional LAPACK wrappers
    subroutine yorghr(n, ilo, ihi, a, lda, tau, work, lwork, info)
        integer(kind=nwchem_integer_kind), intent(in) :: n, ilo, ihi, lda, lwork
        double precision, intent(inout) :: a(lda,*)
        double precision, intent(in) :: tau(*)
        double precision, intent(out) :: work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, ilo_int, ihi_int, lda_int, lwork_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        ilo_int = int(ilo, kind=blas_library_integer_kind)
        ihi_int = int(ihi, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call dorghr(n_int, ilo_int, ihi_int, a, lda_int, tau, work, lwork_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yorghr

    subroutine ypftrf(transr, uplo, n, a, info)
        character(len=1), intent(in) :: transr, uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n
        double precision, intent(inout) :: a(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        call dpftrf(transr, uplo, n_int, a, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ypftrf

    subroutine ypftri(transr, uplo, n, a, info)
        character(len=1), intent(in) :: transr, uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n
        double precision, intent(inout) :: a(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        call dpftri(transr, uplo, n_int, a, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ypftri

    subroutine yposv(uplo, n, nrhs, a, lda, b, ldb, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, lda, ldb
        double precision, intent(inout) :: a(lda,*), b(ldb,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, lda_int, ldb_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        call dposv(uplo, n_int, nrhs_int, a, lda_int, b, ldb_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yposv

    subroutine ypotrf(uplo, n, a, lda, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda
        double precision, intent(inout) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        call dpotrf(uplo, n_int, a, lda_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ypotrf

    subroutine ypotri(uplo, n, a, lda, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda
        double precision, intent(inout) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        call dpotri(uplo, n_int, a, lda_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ypotri

    subroutine ysfrk(transr, uplo, trans, n, k, alpha, a, lda, beta, c)
        character(len=1), intent(in) :: transr, uplo, trans
        integer(kind=nwchem_integer_kind), intent(in) :: n, k, lda
        double precision, intent(in) :: alpha, beta, a(lda,*)
        double precision, intent(inout) :: c(*)
        integer(kind=blas_library_integer_kind) :: n_int, k_int, lda_int
        n_int = int(n, kind=blas_library_integer_kind)
        k_int = int(k, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        call dsfrk(transr, uplo, trans, n_int, k_int, alpha, a, lda_int, beta, c)
    end subroutine ysfrk

    subroutine yspsvx(fact, uplo, n, nrhs, a, afac, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info)
        character(len=1), intent(in) :: fact, uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, ldb, ldx
        double precision, intent(in) :: a(*), b(ldb,*)
        double precision, intent(inout) :: afac(*)
        integer(kind=nwchem_integer_kind), intent(inout) :: ipiv(*)
        double precision, intent(out) :: x(ldx,*), rcond, ferr(*), berr(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: iwork(*), info
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, ldb_int, ldx_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:), iwork_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        ldx_int = int(ldx, kind=blas_library_integer_kind)
        allocate( ipiv_int(n), iwork_int(n) )
        ipiv_int = int(ipiv(1:n), kind=blas_library_integer_kind)
        call dspsvx(fact, uplo, n_int, nrhs_int, a, afac, ipiv_int, b, ldb_int, x, ldx_int, &
                    rcond, ferr, berr, work, iwork_int, info_int)
        ipiv(1:n) = int(ipiv_int(1:n), kind=nwchem_integer_kind)
        deallocate( ipiv_int, iwork_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yspsvx

    subroutine ystebz(range, order, n, vl, vu, il, iu, abstol, d, e, m, nsplit, w, iblock, isplit, work, iwork, info)
        character(len=1), intent(in) :: range, order
        integer(kind=nwchem_integer_kind), intent(in) :: n, il, iu
        double precision, intent(in) :: vl, vu, abstol, d(*), e(*)
        double precision, intent(out) :: w(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: m, nsplit, iblock(*), isplit(*), iwork(*), info
        integer(kind=blas_library_integer_kind) :: n_int, il_int, iu_int, m_int, nsplit_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: iwork_int(:), isplit_int(:), iblock_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        il_int = int(il, kind=blas_library_integer_kind)
        iu_int = int(iu, kind=blas_library_integer_kind)
        allocate( iwork_int(n), isplit_int(n), iblock_int(n) )
        call dstebz(range, order, n_int, vl, vu, il_int, iu_int, abstol, d, e, m_int, nsplit_int, w, iblock_int, isplit_int, work, iwork_int, info_int)
        deallocate( iwork_int, isplit_int, iblock_int )
        m = int(m_int, kind=nwchem_integer_kind)
        nsplit = int(nsplit_int, kind=nwchem_integer_kind)
        iblock(1:n) = int(iblock_int(1:n), kind=nwchem_integer_kind)
        isplit(1:n) = int(isplit_int(1:n), kind=nwchem_integer_kind)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ystebz

    subroutine ystein(n, d, e, m, w, iblock, isplit, z, ldz, work, iwork, ifail, info)
        integer(kind=nwchem_integer_kind), intent(in) :: n, m, ldz
        double precision, intent(in) :: d(*), e(*), w(*)
        integer(kind=nwchem_integer_kind), intent(in) :: iblock(*), isplit(*)
        double precision, intent(out) :: z(ldz,*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: iwork(*), ifail(*), info
        integer(kind=blas_library_integer_kind) :: n_int, m_int, ldz_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: iwork_int(:), isplit_int(:), iblock_int(:), ifail_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        m_int = int(m, kind=blas_library_integer_kind)
        ldz_int = int(ldz, kind=blas_library_integer_kind)
        allocate( iwork_int(n), isplit_int(n), iblock_int(n), ifail_int(m) )
        iblock_int(1:n) = int(iblock(1:n), kind=blas_library_integer_kind)
        isplit_int(1:n) = int(isplit(1:n), kind=blas_library_integer_kind)
        call dstein(n_int, d, e, m_int, w, iblock_int, isplit_int, z, ldz_int, work, iwork_int, ifail_int, info_int)
        deallocate( iwork_int, isplit_int, iblock_int, ifail_int )
        iwork(1:n) = int(iwork_int(1:n), kind=nwchem_integer_kind)
        ifail(1:m) = int(ifail_int(1:m), kind=nwchem_integer_kind)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ystein

    subroutine ysterf(n, d, e, info)
        integer(kind=nwchem_integer_kind), intent(in) :: n
        double precision, intent(inout) :: d(*), e(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        call dsterf(n_int, d, e, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ysterf

    subroutine ysyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info)
        character(len=1), intent(in) :: jobz, uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda, lwork, liwork
        double precision, intent(inout) :: a(lda,*)
        double precision, intent(out) :: w(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: iwork(*), info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, lwork_int, liwork_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: iwork_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        liwork_int = int(liwork, kind=blas_library_integer_kind)
        allocate( iwork_int(liwork) )
        call dsyevd(jobz, uplo, n_int, a, lda_int, w, work, lwork_int, iwork_int, liwork_int, info_int)
        iwork(1:liwork) = int(iwork_int(1:liwork), kind=nwchem_integer_kind)
        deallocate( iwork_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ysyevd

    subroutine ysygv(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info)
        integer(kind=nwchem_integer_kind), intent(in) :: itype, n, lda, ldb, lwork
        character(len=1), intent(in) :: jobz, uplo
        double precision, intent(inout) :: a(lda,*), b(ldb,*)
        double precision, intent(out) :: w(*), work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: itype_int, n_int, lda_int, ldb_int, lwork_int, info_int
        itype_int = int(itype, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call dsygv(itype_int, jobz, uplo, n_int, a, lda_int, b, ldb_int, w, work, lwork_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ysygv

    subroutine ysysv(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, lwork, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, lda, ldb, lwork
        double precision, intent(inout) :: a(lda,*), b(ldb,*)
        double precision, intent(out) :: work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: ipiv(*), info
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, lda_int, ldb_int, lwork_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        allocate( ipiv_int(n) )
        call dsysv(uplo, n_int, nrhs_int, a, lda_int, ipiv_int, b, ldb_int, work, lwork_int, info_int)
        ipiv(1:n) = int(ipiv_int(1:n), kind=nwchem_integer_kind)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ysysv

    ! Additional LAPACK wrappers
    subroutine ytrtri(uplo, diag, n, a, lda, info)
        character(len=1), intent(in) :: uplo, diag
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda
        double precision, intent(inout) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        call dtrtri(uplo, diag, n_int, a, lda_int, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ytrtri

    subroutine ytrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
        character(len=1), intent(in) :: side, uplo, transa, diag
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda, ldb
        double precision, intent(in) :: alpha, a(lda,*)
        double precision, intent(inout) :: b(ldb,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, ldb_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        call dtrmm(side, uplo, transa, diag, m_int, n_int, alpha, a, lda_int, b, ldb_int)
    end subroutine ytrmm

    subroutine ytfsm(transr, side, uplo, trans, unit, m, n, alpha, a, b, ldb)
        character(len=1), intent(in) :: transr, side, uplo, trans, unit
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, ldb
        double precision, intent(in) :: alpha, a(*)
        double precision, intent(inout) :: b(ldb,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, ldb_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        call dtfsm(transr, side, uplo, trans, unit, m_int, n_int, alpha, a, b, ldb_int)
    end subroutine ytfsm

    subroutine ytrevc(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info)
        character(len=1), intent(in) :: side, howmny
        integer(kind=nwchem_integer_kind), intent(in) :: n, ldt, ldvl, ldvr, mm
        logical, intent(in) :: select(*)
        double precision, intent(in) :: t(ldt,*)
        double precision, intent(inout) :: vl(ldvl,*), vr(ldvr,*)
        double precision, intent(out) :: work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: m, info
        integer(kind=blas_library_integer_kind) :: n_int, ldt_int, ldvl_int, ldvr_int, mm_int
        integer(kind=blas_library_integer_kind) :: m_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        ldt_int = int(ldt, kind=blas_library_integer_kind)
        ldvl_int = int(ldvl, kind=blas_library_integer_kind)
        ldvr_int = int(ldvr, kind=blas_library_integer_kind)
        mm_int = int(mm, kind=blas_library_integer_kind)
        call dtrevc(side, howmny, select, n_int, t, ldt_int, vl, ldvl_int, vr, ldvr_int, mm_int, m_int, work, info_int)
        m = int(m_int, kind=nwchem_integer_kind)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine ytrevc

    ! Add more wrapper subroutines...

    ! Complex BLAS Level 1 wrappers
    subroutine yzaxpy(n, alpha, x, incx, y, incy)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx, incy
        complex(kind=8), intent(in) :: alpha, x(*)
        complex(kind=8), intent(inout) :: y(*)
        integer(kind=blas_library_integer_kind) :: n_int, incx_int, incy_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call zaxpy(n_int, alpha, x, incx_int, y, incy_int)
    end subroutine yzaxpy

    subroutine yzcopy(n, x, incx, y, incy)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx, incy
        complex(kind=8), intent(in) :: x(*)
        complex(kind=8), intent(out) :: y(*)
        integer(kind=blas_library_integer_kind) :: n_int, incx_int, incy_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call zcopy(n_int, x, incx_int, y, incy_int)
    end subroutine yzcopy

    subroutine yzdotc(n, x, incx, y, incy, result)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx, incy
        complex(kind=8), intent(in) :: x(*), y(*)
        complex(kind=8), intent(out) :: result
        integer(kind=blas_library_integer_kind) :: n_int, incx_int, incy_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        incy_int = int(incy, kind=blas_library_integer_kind)
        call zdotc(n_int, x, incx_int, y, incy_int, result)
    end subroutine yzdotc

    subroutine yzscal(n, alpha, x, incx)
        integer(kind=nwchem_integer_kind), intent(in) :: n, incx
        complex(kind=8), intent(in) :: alpha
        complex(kind=8), intent(inout) :: x(*)
        integer(kind=blas_library_integer_kind) :: n_int, incx_int
        n_int = int(n, kind=blas_library_integer_kind)
        incx_int = int(incx, kind=blas_library_integer_kind)
        call zscal(n_int, alpha, x, incx_int)
    end subroutine yzscal

    ! Complex BLAS Level 3 wrappers
    subroutine yzgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        character(len=1), intent(in) :: transa, transb
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, k, lda, ldb, ldc
        complex(kind=8), intent(in) :: alpha, beta
        complex(kind=8), intent(in) :: a(lda,*), b(ldb,*)
        complex(kind=8), intent(inout) :: c(ldc,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, k_int, lda_int, ldb_int, ldc_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        k_int = int(k, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        ldc_int = int(ldc, kind=blas_library_integer_kind)
        call zgemm(transa, transb, m_int, n_int, k_int, alpha, a, lda_int, b, ldb_int, beta, c, ldc_int)
    end subroutine yzgemm

    subroutine yzsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
        character(len=1), intent(in) :: uplo, trans
        integer(kind=nwchem_integer_kind), intent(in) :: n, k, lda, ldc
        complex(kind=8), intent(in) :: alpha, beta
        complex(kind=8), intent(in) :: a(lda,*)
        complex(kind=8), intent(inout) :: c(ldc,*)
        integer(kind=blas_library_integer_kind) :: n_int, k_int, lda_int, ldc_int
        n_int = int(n, kind=blas_library_integer_kind)
        k_int = int(k, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldc_int = int(ldc, kind=blas_library_integer_kind)
        call zsyrk(uplo, trans, n_int, k_int, alpha, a, lda_int, beta, c, ldc_int)
    end subroutine yzsyrk

    ! Complex LAPACK wrappers
    subroutine yzgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info)
        character(len=1), intent(in) :: jobvl, jobvr
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda, ldvl, ldvr, lwork
        complex(kind=8), intent(inout) :: a(lda,*)
        complex(kind=8), intent(out) :: w(*), vl(ldvl,*), vr(ldvr,*), work(*)
        double precision, intent(out) :: rwork(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, ldvl_int, ldvr_int, lwork_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldvl_int = int(ldvl, kind=blas_library_integer_kind)
        ldvr_int = int(ldvr, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call zgeev(jobvl, jobvr, n_int, a, lda_int, w, vl, ldvl_int, vr, ldvr_int, work, lwork_int, rwork, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yzgeev

    subroutine yzgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info)
        character(len=1), intent(in) :: jobu, jobvt
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda, ldu, ldvt, lwork
        complex(kind=8), intent(inout) :: a(lda,*)
        double precision, intent(out) :: s(*)
        complex(kind=8), intent(out) :: u(ldu,*), vt(ldvt,*), work(*)
        double precision, intent(out) :: rwork(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, ldu_int, ldvt_int, lwork_int, info_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldu_int = int(ldu, kind=blas_library_integer_kind)
        ldvt_int = int(ldvt, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call zgesvd(jobu, jobvt, m_int, n_int, a, lda_int, s, u, ldu_int, vt, ldvt_int, work, lwork_int, rwork, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yzgesvd

    subroutine yzgetrf(m, n, a, lda, ipiv, info)
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda
        complex(kind=8), intent(inout) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(out) :: ipiv(*), info
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        allocate( ipiv_int(min(m,n)) )
        call zgetrf(m_int, n_int, a, lda_int, ipiv_int, info_int)
        ipiv(1:min(m,n)) = int(ipiv_int(1:min(m,n)), kind=nwchem_integer_kind)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yzgetrf

    subroutine yzgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
        character(len=1), intent(in) :: trans
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, lda, ldb
        complex(kind=8), intent(in) :: a(lda,*)
        integer(kind=nwchem_integer_kind), intent(in) :: ipiv(*)
        complex(kind=8), intent(inout) :: b(ldb,*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, lda_int, ldb_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        allocate( ipiv_int(n) )
        ipiv_int(1:n) = int(ipiv(1:n), kind=blas_library_integer_kind)
        call zgetrs(trans, n_int, nrhs_int, a, lda_int, ipiv_int, b, ldb_int, info_int)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yzgetrs

    subroutine yzheev(jobz, uplo, n, a, lda, w, work, lwork, rwork, info)
        character(len=1), intent(in) :: jobz, uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda, lwork
        complex(kind=8), intent(inout) :: a(lda,*)
        double precision, intent(out) :: w(*)
        complex(kind=8), intent(out) :: work(*)
        double precision, intent(out) :: rwork(*)
        integer(kind=nwchem_integer_kind), intent(out) :: info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, lwork_int, info_int
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        call zheev(jobz, uplo, n_int, a, lda_int, w, work, lwork_int, rwork, info_int)
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yzheev

    subroutine yzlacp2(uplo, m, n, a, lda, b, ldb)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda, ldb
        double precision, intent(in) :: a(lda,*)
        complex(kind=8), intent(out) :: b(ldb,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, ldb_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        call zlacp2(uplo, m_int, n_int, a, lda_int, b, ldb_int)
    end subroutine yzlacp2

    subroutine yzlacpy(uplo, m, n, a, lda, b, ldb)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: m, n, lda, ldb
        complex(kind=8), intent(in) :: a(lda,*)
        complex(kind=8), intent(out) :: b(ldb,*)
        integer(kind=blas_library_integer_kind) :: m_int, n_int, lda_int, ldb_int
        m_int = int(m, kind=blas_library_integer_kind)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        call zlacpy(uplo, m_int, n_int, a, lda_int, b, ldb_int)
    end subroutine yzlacpy

    subroutine yzsysv(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, lwork, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, nrhs, lda, ldb, lwork
        complex(kind=8), intent(inout) :: a(lda,*), b(ldb,*)
        complex(kind=8), intent(out) :: work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: ipiv(*), info
        integer(kind=blas_library_integer_kind) :: n_int, nrhs_int, lda_int, ldb_int, lwork_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        nrhs_int = int(nrhs, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        ldb_int = int(ldb, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        allocate( ipiv_int(n) )
        call zsysv(uplo, n_int, nrhs_int, a, lda_int, ipiv_int, b, ldb_int, work, lwork_int, info_int)
        ipiv(1:n) = int(ipiv_int(1:n), kind=nwchem_integer_kind)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yzsysv

    subroutine yzsytrf(uplo, n, a, lda, ipiv, work, lwork, info)
        character(len=1), intent(in) :: uplo
        integer(kind=nwchem_integer_kind), intent(in) :: n, lda, lwork
        complex(kind=8), intent(inout) :: a(lda,*)
        complex(kind=8), intent(out) :: work(*)
        integer(kind=nwchem_integer_kind), intent(out) :: ipiv(*), info
        integer(kind=blas_library_integer_kind) :: n_int, lda_int, lwork_int, info_int
        integer(kind=blas_library_integer_kind), allocatable :: ipiv_int(:)
        n_int = int(n, kind=blas_library_integer_kind)
        lda_int = int(lda, kind=blas_library_integer_kind)
        lwork_int = int(lwork, kind=blas_library_integer_kind)
        allocate( ipiv_int(n) )
        call zsytrf(uplo, n_int, a, lda_int, ipiv_int, work, lwork_int, info_int)
        ipiv(1:n) = int(ipiv_int(1:n), kind=nwchem_integer_kind)
        deallocate( ipiv_int )
        info = int(info_int, kind=nwchem_integer_kind)
    end subroutine yzsytrf

end module blas
