      subroutine dfill(n,val,a,ia)
      implicit none
      integer n, ia
      double precision val, a(n)
      call dcopy(n,val,0,a,ia)
      end

      subroutine ccsd_aosig(nocc,nvir,nbf,cmo,scra,scrb)
      implicit none
      integer nocc,nvir,nbf
      double precision cmo(nbf,nbf)
      double precision scra(nbf,nbf)
      double precision scrb(nbf*nbf)
      integer p,q,a,b
      call dfill(nbf*nbf,0.0d00,scra,1)
      do p=1,nbf
       do q=1,nbf
        do a=1,nvir
         do b=1,nvir
          scra(p,q) = scra(p,q) + scrb((b-1) * nvir+a) * cmo(p,nocc+a)
     &              * cmo(q,nocc+b)
         enddo
        enddo
       enddo
      enddo
      end subroutine

      subroutine new_aosig(nocc,nvir,nbf,cmo,scra,scrb)
      implicit none
      integer nocc,nvir,nbf
      double precision cmo(nbf,nbf)
      double precision scra(nbf,nbf)
      double precision scrb(nvir,*)
      integer p,q,a,b
      !call dfill(nbf*nbf,0.0d00,scra,1)
      scra = 0.0d0
      do p=1,nbf
       do q=1,nbf
        do a=1,nvir
         do b=1,nvir
          !print*,'off=',a,b,nvir,a+(b-1)*nvir
          scra(p,q) = scra(p,q) + scrb(a,b) * cmo(p,nocc+a)
     &              * cmo(q,nocc+b)
         enddo
        enddo
       enddo
      enddo
      end subroutine

      program main
      implicit none
      integer nocc,nvir,nbf
      parameter (nbf = 117)
      double precision cmo(nbf,nbf)  ! input
      double precision inp(nbf,nbf)  ! input
      double precision out1(nbf,nbf) ! in/out 1
      double precision out2(nbf,nbf) ! in/out 2
      nocc = nbf/5
      if (nocc.lt.2) nocc = 2
      nvir = nbf - nocc
      if (nvir.lt.2) stop
      print*,'nbf=',nbf
      print*,'nocc=',nocc
      print*,'nvir=',nvir
      call random_number(cmo)
      call random_number(inp)
      print*,'ccsd_aosig'
      call ccsd_aosig(nocc,nvir,nbf,cmo,out1,inp)
      print*,'new_aosig'
      call new_aosig(nocc,nvir,nbf,cmo,out2,inp)
      print*,'correct:',all(out1.eq.out2)
      if (nbf.lt.8) then
        print*,out1
        print*,out2
      endif
      end program main
