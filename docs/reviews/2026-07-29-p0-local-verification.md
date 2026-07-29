# P0 local verification

The isolated test mirror was verified with:

```text
PYTHONPATH=src pytest -q
................                                                         [100%]
16 passed in 0.99s

python -m compileall -q src scripts tests
exit status: 0
```

The mirror contains the same production and test files changed by this branch. A full dependency-backed model instantiation was not run in the isolated environment because `transformers` is not installed there; the repository requirements include it for CI and normal development environments.
