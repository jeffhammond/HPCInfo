# Basics

```
perf record <binary> <arguments>
```
Weird things happen when I pipe this to `tee`, so remember to `>& log &` then `tail -f log` instead.
```
perf report
```
Why can't I use VI-style page-up/down here?  Blows my mind.