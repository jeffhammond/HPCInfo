# NaN handling in a MAX operator: review of the arguments

The evidence is mostly standards-committee papers and compiler-developer discussions. I found little peer-reviewed numerical-analysis literature that argues the question directly. N2532, the main C proposal, was a compressed PDF that I could not read. Its content below comes from search summaries and from the P3008 and LLVM threads that cite it. I could not retrieve the IEEE 754-2019 text, so check normative details there.

## The two semantics

| Semantics | `max(x, NaN)` | IEEE 754-2019 | C23 | Mental model |
|---|---|---|---|---|
| **Propagate** | NaN | `maximum` | `fmaximum` | NaN means an error |
| **Ignore** | `x` | `maximumNumber` | `fmaximum_num` (and `fmax`) | NaN means missing data |

P3008 uses the same "Error" versus "Missing Data" framing. The ignore rule returns NaN only when every input is NaN.

## Arguments for propagating NaN

- **NaN is a poisoned result.** An upstream 0/0, ∞−∞ or overflow means the computation went wrong. A max that silently returns a finite value hides that, and the result looks plausible but is wrong.
- **It matches the rest of IEEE arithmetic.** `+`, `*`, `sqrt` and similar operations all propagate NaN, so one NaN input contaminates the output. Convergence tests such as `max(abs(r)) < tol` then fail loudly, where an ignoring max can let them pass. Linear-algebra norms and residual checks depend on this.
- **It is order-independent and associative.** With propagation, `max(a, max(b, c))` equals `max(max(a, b), c)`, so parallel and vectorized reductions give the same answer however they split the work. This is the central argument for the 2019 revision. Stephen Canon says the old `minNum`/`maxNum` were removed from the normative clauses mainly because they are not associative. The C proposal cites the same reason.
- **Hardware supports it.** ARMv8 `FMAX` propagates NaN. The naive `a > b ? a : b` form does not, because NaN comparisons are false and the result depends on operand order.
- **Users can opt in.** P3008 proposes explicit `fetch_fmaximum` for people who want error semantics.

## Arguments for ignoring NaN

- **NaN often means missing data.** In statistics, data analysis and sensor or table processing, NaN marks an absent entry. The max of the valid entries is what the user wants. MATLAB `max` defaults to `omitnan`, and ILNumerics returns NaN only if every element is NaN.
- **It is the long-standing C and POSIX behavior.** C99 `fmax` returns the non-NaN operand, and C23 note 299 says quiet NaNs are "treated as missing data."
- **Hardware and performance.** P3008 says every GPU architecture it surveyed treats quiet NaNs as missing data in native atomic min/max. This covers AMD CDNA2+, Intel Xe, NVIDIA PTX and the SPIR-V extension. Native atomics can be orders of magnitude faster than compare-and-swap loops. On AMDGPU, LLVM had to insert `canonicalize` instructions to get sNaN-correct behavior, which roughly tripled the instruction count in one kernel's inner loop.
- **Compatibility.** A large amount of existing code relies on `fmax` ignoring NaN.

## Complications

- **Signaling NaNs break the ignore-NaN operation.** IEEE 754-2008 `maxNum(1, sNaN)` returns a quiet NaN, while C `fmax` returns 1. This mismatch (LLVM issue 27363, Arsenault's 2018 RFC) is what made `maxNum` non-associative in the first place. One example: `maxNum(maxNum(sNaN, 1), 2)` gives 2, while `maxNum(sNaN, maxNum(1, 2))` gives NaN. Canon's advice at the time was not to implement the replacements yet because they were still non-normative. A 2024 post reports that glibc `fmin`/`fmax` have returned NaN for a signaling NaN since 2021, which may conflict with C23.
- **Signed zeros matter as much.** Both 2019 families order −0 before +0, and Canon considers this possibly more important than the sNaN behavior. x86 `MAXSD` returns the second operand when the inputs compare equal, which loses that ordering.
- **Hardware is split.** x86 `MAXSD`/`MINSD` return the second operand on any NaN, which is the naive ternary behavior. ARM provides both `FMAX` (propagate) and `FMAXNM` (ignore).
- **The committee did not pick one.** IEEE 754-2019 provides both families, and C23 follows with two function sets. That suggests no single default suits every use. In the P3008 review, WG21's SG6 chose to leave the NaN result unspecified for the default atomic `fetch_max`. One reviewer called that "bad for teachability."

## Where the arguments leave it

- **Propagate** for reductions in numerical methods, such as norms, residuals and convergence checks, and for anything that must be reproducible under reordering. Here a NaN signals a bug that should surface.
- **Ignore** for data handling with real missing values, and where legacy `fmax` or hardware atomics are involved. Handle sNaN and signed-zero cases explicitly when the code must be bit-reproducible.

Mixing the two in one code base is the main hazard. That includes naive ternary max, `fmax` and Fortran `MAX`, whose NaN behavior I did not verify. The test in `nan_minmax.c` showed `fmax` ignoring NaN and the ternary form depending on operand order.

## Not verified

I relied on memory for the Julia and Fortran claims, which the search did not cover. Julia's `maximum` propagates NaN, and Fortran's `MAX` with a NaN is processor-dependent. The ILNumerics page described numpy as propagating, and that is secondhand. I also found no primary IEEE rationale and no sources on Kahan's views. For a paper, the next step is IEEE 754-2019 §9.6, the WG14 papers N2532 and N2489, and Kahan's writing on NaN semantics.

## Sources

- [WG14 N2489](https://www9.open-std.org/jtc1/sc22/wg14/www/docs/n2489.pdf)
- [WG14 N2532, "Min-max functions"](https://www9.open-std.org/jtc1/sc22/wg14/www/docs/n2532.pdf)
- [P3008R3, "Atomic floating-point min/max"](https://open-std.org/JTC1/SC22/WG21/docs/papers/2024/p3008r3.html)
- [LLVM discourse: "What is the real behavior for the minnum/maxnum intrinsics?"](https://discourse.llvm.org/t/rfc-what-is-the-real-behavior-for-the-minnum-maxnum-intrinsics/49211)
- [LLVM bug 27363: @llvm.maxnum.* and signaling NaNs](https://lists.llvm.org/pipermail/llvm-bugs/2016-April/046302.html)
- [Wikipedia: NaN](https://en.wikipedia.org/wiki/NaN)
- [MATLAB `max` documentation](https://cs400-web.cs.wisc.edu/nchou/s/matlab-2020b/amd64_ubu22/help/matlab/ref/max.html)
- [MathWorks forum: sum and max NaN defaults](https://la.mathworks.com/matlabcentral/answers/778987-default-sum-and-max-for-nan-is-not-consistent)
- [ILNumerics `maxall`](https://ilnumerics.net/apidoc/html/M_ILNumerics_ILMath_maxall_3.htm)
