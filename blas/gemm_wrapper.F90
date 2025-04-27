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

        ! BLAS Level 3
        subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            character(len=1), intent(in) :: transa, transb
            integer(kind=blas_library_integer_kind), intent(in) :: m, n, k, lda, ldb, ldc
            double precision, intent(in) :: alpha, beta
            double precision, intent(in) :: a(lda,*), b(ldb,*)
            double precision, intent(out) :: c(ldc,*)
        end subroutine dgemm

    end interface

    contains

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

end module blas
