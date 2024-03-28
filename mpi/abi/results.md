# MPICH with Fortran support

% /opt/mpich/debug-ofi/bin/mpirun -n 1 ./c-logical-reduction.x ; /opt/mpich/debug-ofi/bin/mpirun -n 1 ./f-logical-reduction.x
C time=0.069964 (for 1000000 calls)
OK
F time=   7.1035999999992328E-002  (for 1000000 calls)
OK

# MPICH without Fortran support, plus VAPAA

C time=0.072256 (for 1000000 calls)
OK
F time=   2.6780559999999696       (for 1000000 calls)
OK

# Mukautuva using MPICH without Fortran support, plus VAPAA

C time=0.174817 (for 1000000 calls)
OK
F time=  0.15778199999999742       (for 1000000 calls)
OK
