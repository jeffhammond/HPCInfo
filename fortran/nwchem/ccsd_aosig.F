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
      do q=1,nbf
       do a=1,nvir
        do b=1,nvir
         do p=1,nbf
          scra(p,q) = scra(p,q) + scrb(a,b) * cmo(p,nocc+a)
     &              * cmo(q,nocc+b)
         enddo
        enddo
       enddo
      enddo
      end subroutine

      subroutine new2_aosig(nocc,nvir,nbf,cmo,scra,scrb)
      implicit none
      integer nocc,nvir,nbf
      double precision cmo(nbf,nbf)
      double precision cmo2(nbf,nvir)
      double precision scra(nbf,nbf)
      double precision scrb(nvir,nvir)
      integer p,q,a,b
      cmo2 = cmo(:,nocc+1:nbf)
      scra = 0.0d0
      do q=1,nbf
       do a=1,nvir
        do b=1,nvir
         do p=1,nbf
          scra(p,q) = scra(p,q) + scrb(a,b) * cmo2(p,a) * cmo2(q,b)
         enddo
        enddo
       enddo
      enddo
      end subroutine


      subroutine new3_aosig(nocc,nvir,nbf,cmo,scra,scrb)
      implicit none
      integer nocc,nvir,nbf
      double precision cmo(nbf,nbf)
      double precision cmo2(nbf,nvir)
      double precision cmo2t(nvir,nbf)
      double precision scra(nbf,nbf)
      double precision scrb(nvir,nvir)
      !double precision z(nvir,nvir)
      !double precision :: sab
      double precision :: x,y
      integer p,q,a,b
      cmo2 = cmo(:,nocc+1:nbf)
      cmo2t = transpose(cmo2)
      scra = 0.0d0
      !z = 0.0d0
      do q=1,nbf
       do a=1,nvir
        do b=1,nvir
         y = scrb(a,b)
         do p=1,nbf
          !scra(p,q) = scra(p,q) + cmo2(p,a) * (scrb(a,b) * cmo2(q,b)) ! 1e-11 error
          !scra(p,q) = scra(p,q) + (cmo2(p,a) * scrb(a,b)) * cmo2(q,b)
          !scra(p,q) = scra(p,q) + (cmo2(p,a) * scrb(a,b)) * x
          !scra(p,q) = scra(p,q) + (cmo2(p,a) * y) * x ! OK
          !scra(p,q) = scra(p,q) + (cmo2(p,a) * y) * cmo2(q,b) ! fast w/ qabp order
          scra(p,q) = scra(p,q) + (cmo2(p,a) * y) * cmo2t(b,q)
          !sab = scrb(a,b)
          !scra(p,q) = scra(p,q) + sab * cmo2(p,a) * cmo2(q,b)
         enddo
        enddo
       enddo
      enddo
      end subroutine


      program main
      implicit none
      integer nocc,nvir,nbf
      parameter (nbf = 200)
      double precision cmo(nbf,nbf)  ! input
      double precision inp(nbf,nbf)  ! input
      double precision out1(nbf,nbf) ! in/out 1
      double precision out2(nbf,nbf) ! in/out 2
      double precision out3(nbf,nbf) ! in/out 3
      double precision out4(nbf,nbf) ! in/out 4
      real :: ts,t0,t1,t2,t3
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
      call cpu_time(ts)
      call ccsd_aosig(nocc,nvir,nbf,cmo,out1,inp)
      call cpu_time(t0)
      t0 = t0 - ts
      print*,'t0=',t0

      print*,'new_aosig'
      call new_aosig(nocc,nvir,nbf,cmo,out2,inp)
      call cpu_time(ts)
      call new_aosig(nocc,nvir,nbf,cmo,out2,inp)
      call cpu_time(t1)
      t1 = t1 - ts
      print*,'t1=',t1

      print*,'new2_aosig'
      call new2_aosig(nocc,nvir,nbf,cmo,out3,inp)
      call cpu_time(ts)
      call new2_aosig(nocc,nvir,nbf,cmo,out3,inp)
      call cpu_time(t2)
      t2 = t2 - ts
      print*,'t2=',t2

      print*,'new3_aosig'
      call new3_aosig(nocc,nvir,nbf,cmo,out4,inp)
      call cpu_time(ts)
      call new3_aosig(nocc,nvir,nbf,cmo,out4,inp)
      call cpu_time(t3)
      t3 = t3 - ts
      print*,'t3=',t3

      print*,'new  correct:',all((out1-out2).lt.1e-11)
      print*,'err  = ',norm2(out1-out2)
      print*,'new2 correct:',all((out1-out3).lt.1e-11)
      print*,'err2 = ',norm2(out1-out3)
      print*,'new3 correct:',all((out1-out4).lt.1e-11)
      print*,'err3 = ',norm2(out1-out4)

      if (nbf.lt.8) then
        print*,out1
        print*,out2
        print*,out3
        print*,out4
      endif

      end program main
