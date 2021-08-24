      subroutine dfill(n,val,a,ia)
      implicit none
      integer n, ia
      double precision val, a(*)
      integer i
      call dcopy(n,val,0,a,ia)
      end

      subroutine ccsd_aosig(nocc,nvir,nbf,cmo,scra,scrb)
      implicit none
      integer nocc,nvir,nbf
      double precision cmo(nbf,nbf),scra(nbf,nbf),scrb(nbf*nbf)
      integer p,q,a,b

      call dfill(nbf*nbf,0.0d00,scra,1)
      do p=1,nbf
       do q=1,nbf
        do a=1,nvir
         do b=1,nvir
        scra(p,q) = scra(p,q) + scrb((b-1) * nvir+a) * cmo(p,nocc+a)
     &            * cmo(q,nocc+b)
         enddo
        enddo
       enddo
      enddo
      end subroutine

      program main


      end program main
