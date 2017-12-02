# Systems

## Cray Red Storm and XT (SeaStar interconnect)

The Cray XT series used the Seastar interconnect and Portals 3 network API, which were designed in collaboration with Sandia National Laboratory to be efficient for [MPI](https://github.com/jeffhammond/HPCInfo/blob/master/mpi) programs.  Seastar connects directly to the CPUs via HyperTransport, which limits the CPU sockets to those produced by AMD.

* [Cray XT4: an early evaluation for petascale scientific simulation](http://dx.doi.org/10.1145/1362622.1362675)
* [The Cray XT4 and Seastar 3-D Torus Interconnect](http://research.google.com/pubs/archive/36896.pdf)
* [Jaguar: The World’s Most Powerful Computer](http://www.nccs.gov/wp-content/uploads/2010/01/Bland-Jaguar-Paper.pdf)

## Cray XE (Gemini interconnect)

The Cray XE series introduced the Gemini interconnect, which has much better support for one-sided programming models, including [SHMEM](https://github.com/jeffhammond/HPCInfo/blob/master/shmem), [UPC](https://github.com/jeffhammond/HPCInfo/blob/master/upc) and [CAF](https://github.com/jeffhammond/HPCInfo/blob/master/coarray-f), using the [DMAPP](https://github.com/jeffhammond/HPCInfo/blob/master/dmapp) API.  [MPI](https://github.com/jeffhammond/HPCInfo/blob/master/mpi) performance improved substantially in the area of messaging-rate despite the lack of hardware-assisted mapping.

Gemini connects directly to the CPUs via HyperTransport, which limits the CPU sockets to those produced by AMD.

* [The Gemini System Interconnect](http://dx.doi.org/10.1109/HOTI.2010.23)
* [Investigating the Impact of the Cielo Cray XE6 Architecture on Scientific Application Codes](http://dx.doi.org/10.1109/IPDPS.2011.342)
* [A preliminary evaluation of the hardware acceleration of the cray gemini interconnect for PGAS languages and comparison with MPI](http://doi.acm.org/10.1145/2088457.2088467)
* [A uGNI-based Asynchronous Message-driven Runtime System for Cray Supercomputers with Gemini Interconnect](http://dx.doi.org/10.1109/IPDPS.2012.127)
* [DMAPP - An API for One-sided Program Models on Baker Systems](https://cug.org/5-publications/proceedings_attendee_lists/CUG10CD/pages/1-program/final_program/CUG10_Proceedings/pages/authors/01-5Monday/03B-tenBruggencate-Paper-2.pdf)

### Random Notes

Modules I need to load:
* ```module load moab torque rca pmi cce PrgEnv-pgi ugni xtpe-mc12 xtpe-network-gemini```

Topology information (from Abhinav Bhatele):
* ```xtnodestat```
* ```xtprocadmin```
* ```mysql -h sdb -B --disable-column-names -D XTAdmin -e 'select processor_id,x_coord,y_coord,z_coord from processor where processor_type='\''compute'\'' and processor_status#'\''up'\'''```

Places to look for useful system headers:
* ```/opt/cray/pmi/default/include```
* ```/opt/cray/rca/default/include```
* ```/opt/cray-hss-devel/default/include/rsms/```

## Cray XK (Gemini interconnect)

The Cray XK series is similar to the Cray XE except for the presence of GPUs.  It is the first CPU-GPU architecture from Cray.

* [An Evaluation of Molecular Dynamics Performance on the Hybrid Cray XK6 Supercomputer](http://www.sciencedirect.com/science/article/pii/S187705091200141X)

## Cray XC (Aries interconnect)

The Cray XC series is the implementation of the DARPA-funded [Cascade](http://www.cray.com/Programs/Cascade.aspx) architecture, which includes the Cray Aries interconnect.  Unlike the Seastar and Gemini networks, which had a 3D torus topology, the Aries network is a Dragonly topology.  Additionally, Aries talks to the CPU via PCI, which permits both AMD and Intel CPUs, although all known installations are Intel-based.

NERSC [Edison](http://www.nersc.gov/users/computational-systems/edison/) and CSCS [Piz Daint] (http://www.cscs.ch/fileadmin/user_upload/customers/CSCS_Application_Data/Events/USER_DAY/UD_20-21_Sep_2012/UserDay2012Stringfellow.pdf) are two of the first XC machines.

* [Cray Cascade: a Scalable HPC System based on a Dragonfly Network](http://conferences.computer.org/sc/2012/papers/1000a079.pdf)
* [Cray High Speed Networking](http://www.hoti.org/hoti20/slides/Bob_Alverson.pdf)
* [Cray XC Series Network](https://www.nersc.gov/assets/Uploads/CrayXC30Networking.pdf)
* [NERSC Edison information](http://www.nersc.gov/users/NUG/annual-meetings/2013/nug-2013-training-edison/)

### Topology

#### Background

Thanks to Mike Stewart of NERSC and Dave Strenski of Cray for helping me figure this out.

The following is verbatim from Stewart:
```
To know the physical node location you need to use 'xtprocadmin' or see
/etc/hosts where Cray uses a notation like c12-5c2s7n1, which means the
column 12, row 5, cage 2, slot 7 (blade), node 1.
```

The following is verbatim from Strenski.
```
If you just looking for proximity, then the cname will tell you.
For example.

    c12-5c2s7n0    c12-5c2s7n1    c12-5c2s7n2    c12-5c2s7n3

are all on the same blade connected to the same Aries chip. This
is the closest/fastest connection between nodes.

The next "layer" would be at the chassis level

     c12-5c2s0n[0,1,2,3]              c12-5c2s1n[0,1,2,3]
     c12-5c2s2n[0,1,2,3]              c12-5c2s3n[0,1,2,3]
     c12-5c2s4n[0,1,2,3]              c12-5c2s5n[0,1,2,3]
     c12-5c2s6n[0,1,2,3]              c12-5c2s7n[0,1,2,3]
     c12-5c2s8n[0,1,2,3]              c12-5c2s9n[0,1,2,3]
     c12-5c2s10n[0,1,2,3]             c12-5c2s11n[0,1,2,3]
     c12-5c2s12n[0,1,2,3]             c12-5c2s13n[0,1,2,3]
     c12-5c2s14n[0,1,2,3]             c12-5c2s15n[0,1,2,3]

All of these nodes talk to an aries chip, through a backplane, to
a second aries chip and to the node. This is an all-to-all network.

The next layer is called a group consisting of 6 chassis in two
cabinets.

    c12-5c0s[0,1,2,3,4,5]n[0,1,2,3]
    c12-5c1s[0,1,2,3,4,5]n[0,1,2,3]
    c12-5c2s[0,1,2,3,4,5]n[0,1,2,3]
    c12-5c3s[0,1,2,3,4,5]n[0,1,2,3]
    c12-5c4s[0,1,2,3,4,5]n[0,1,2,3]
    c12-5c5s[0,1,2,3,4,5]n[0,1,2,3]

This is also an all-to-all connection connecting aries in groups of 6.
```

For my own reference, this is what Edison looks like right now (Phase 1):
```
jhammond@edison04:/opt/cray> xtprocadmin
NID (HEX) NODENAME TYPE STATUS MODE
1 0x1 c0-0c0s0n1 service up batch
2 0x2 c0-0c0s0n2 service up batch
5 0x5 c0-0c0s1n1 service up batch
6 0x6 c0-0c0s1n2 service up batch
8 0x8 c0-0c0s2n0 service up other
9 0x9 c0-0c0s2n1 service up other
10 0xa c0-0c0s2n2 service up other
...
760 0x2f8 c3-0c2s14n0 compute up batch
761 0x2f9 c3-0c2s14n1 compute up batch
762 0x2fa c3-0c2s14n2 compute up batch
763 0x2fb c3-0c2s14n3 compute up batch
764 0x2fc c3-0c2s15n0 compute up batch
765 0x2fd c3-0c2s15n1 compute up batch
766 0x2fe c3-0c2s15n2 compute up batch
767 0x2ff c3-0c2s15n3 compute up batch
```

Not from the logins, but inside of a job (including an interactive one), one can get the aforementioned information from the following files:
```
jhammond@nid00265:> ls /proc/cray_xt/
cname  nid
jhammond@nid00265:> ls /proc/cray_xt/cname 
/proc/cray_xt/cname
jhammond@nid00265:> more /proc/cray_xt/cname 
c1-0c1s2n1
jhammond@nid00265:> more /proc/cray_xt/nid 
265
```

#### XC Topology

See [xctopo.h](https://github.com/jeffhammond/HPCInfo/blob/master/topology/xctopo.h), [xctopo.c](https://github.com/jeffhammond/HPCInfo/blob/master/topology/xctopo.c), and [test.c](https://github.com/jeffhammond/HPCInfo/blob/master/topology/test.c)

# System Software

Cray provides an excellent implementation of the [SHMEM](https://github.com/jeffhammond/HPCInfo/blob/master/shmem/README.md) runtime as well as optimized [UPC](https://github.com/jeffhammond/HPCInfo/blob/master/upc/README.md) and [CAF](https://github.com/jeffhammond/HPCInfo/blob/master/coarray-f/README.md) compilers on all of their systems.

## DMAPP

See [DMAPP](https://github.com/jeffhammond/HPCInfo/blob/master/dmapp).  This API is available on the Cray XE, XK and XC architectures, but not the XT architecture.

If you want MPI RMA to use DMAPP, set `MPICH_RMA_OVER_DMAPP=1` in your job environment.  See [`man intro_mpi`](http://docs.cray.com/cgi-bin/craydoc.cgi?mode=View;id=sw_releases-o23alcrv-1426185385;idx=man_search;this_sort=release_date%20desc;q=MPICH_RMA_OVER_DMAPP;type=man;title=Message%20Passing%20Toolkit%20%28MPT%29%207.2%20Man%20Pages) for details.

## Topology

See [cray.c](https://github.com/jeffhammond/HPCInfo/blob/master/topology/cray.c) in the repo for older info.

## More Linux on the Compute Nodes

Running ```make check``` interactively sometimes requires shell commands that aren't loaded by default in the stripped-down Cray Linux Environment.  The following option resolves this.

```
export CRAY_ROOTFS=DSL
```