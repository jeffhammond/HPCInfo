# Events

These files have lists of supported events:
```
ls -l /opt/intel/vtune_amplifier_xe/config/sampling/*
```

Alternatively, one can query SEP:
```
sep -el
```

# Examples
```
amplxe-cl -collect-with runsa \
-knob event-config=CPU_CLK_UNHALTED.CORE,CPU_CLK_UNHALTED.REF,INST_RETIRED.ANY,\
MEM_INST_RETIRED.ALL_LOADS,MEM_INST_RETIRED.ALL_STORES,\
FP_ARITH_INST_RETIRED.SCALAR_DOUBLE,FP_ARITH_INST_RETIRED.128B_PACKED_DOUBLE,\
FP_ARITH_INST_RETIRED.256B_PACKED_DOUBLE,FP_ARITH_INST_RETIRED.512B_PACKED_DOUBLE \
-- $(PROG) $(ARGS)
```