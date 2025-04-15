Slurm is dumb and won't do the right thing by default...

# LUMI

```
> salloc --mem=256G --nodes=2 --time=00:60:00 --ntasks-per-node 64 \
         --cpus-per-task=1 --partition=small
```

Check the allocation makes sense with `srun hostname`.

```
~/NWCHEM/jobs> ARMCI_USE_REQUEST_ATOMICS=0 ARMCI_RMA_ATOMICITY=0 ARMCI_VERBOSE=1 \
               srun -N 2 --ntasks-per-node 32 \
               /users/jhammond/NWCHEM/git/bin/LINUX64/nwchem \
               w7_rccsd-t_cc-pvdz_energy.nw 2>&1 | tee w7_rccsd-t_cc-pvdz_energy.log.9
```
