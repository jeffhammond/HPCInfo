program main
    use, intrinsic ::  iso_c_binding
    use, intrinsic :: iso_fortran_env
    print*,kind('A'),kind('€'),kind('🤓')
    block
      character(1) :: a
      character(kind=1) :: b
      character(kind=kind('A')) :: c = 'A'
      character(kind=kind('€')) :: d = '€'
      character(kind=kind('🤓')) :: e = '🤓'
      print*,storage_size(a)
      print*,storage_size(b)
      print*,storage_size(c)
      print*,storage_size(d)
      print*,storage_size(e)
      print*,c,d,e
    end block
#if 0
    block
      integer,parameter :: ucs4 = selected_char_kind("ISO_10646")
      character(kind=ucs4) :: c = 'A'
      character(kind=ucs4) :: d = '€'
      character(kind=ucs4) :: e = '🤓'
      print*,storage_size(c)
      print*,storage_size(d)
      print*,storage_size(e)
      print*,c,d,e
    end block
#endif
    block
      INTRINSIC date_and_time,selected_char_kind
      INTEGER,PARAMETER :: ucs4 = selected_char_kind("ISO_10646")
      CHARACTER(1,UCS4),PARAMETER :: nen=CHAR(INT(Z'5e74'),UCS4), & !year
                                   gatsu=CHAR(INT(Z'6708'),UCS4), & !month
                                   nichi=CHAR(INT(Z'65e5'),UCS4)    !day
      INTEGER values(8)
      CALL date_and_time(values=values)
      WRITE(*,1) values(1),nen,values(2),gatsu,values(3),nichi
    1 FORMAT(I0,A,I0,A,I0,A)
    end block
end program main
