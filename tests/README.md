# `tests`

| file | what it does |
|---|---|
| `test_pipeline.py` | End-to-end validation on synthetic jets, about 2 minutes on a CPU. |

It runs three groups of checks:
1. **Unit checks:** fixed-point rounding and saturation, config round trip
   and typo rejection, ROOT feature building and book-keeping on in-memory
   awkward arrays.
2. **The README pipeline:** every command-line step, in order, on a toy
   dataset. It checks that each script exits cleanly and writes its figures.
3. **Physics sanity checks:**
   - the model out-resolves the binned median;
   - the binned median closes;
   - 24-bit PTQ reproduces the float model;
   - the PFN is permutation invariant;
   - Expected-Gradients completeness is below 2%.

```bash
python tests/test_pipeline.py              # temporary directory, cleaned up
python tests/test_pipeline.py --keep out   # keep every output for inspection
```

It prints one `PASS`/`FAIL` line per check, stops at the first failure, and
ends with `ALL <n> CHECKS PASSED`.
