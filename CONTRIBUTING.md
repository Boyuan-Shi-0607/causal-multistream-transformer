# Contributing

Thanks for helping improve Causal Multistream Transformer. Focused bug fixes,
portability improvements, documentation, and synthetic correctness tests are
especially useful.

## Development setup

Use a C++17 compiler, CMake, and a Python environment containing PyTorch. The
currently validated dependency is PyTorch 2.9.1.

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/cmt_synthetic
```

## Change guidelines

- Keep the public API task-agnostic and contained in namespace `cmt`.
- Preserve the separation between RoPE position and causal availability.
- Add or update a synthetic test for behavioral changes.
- Keep tests deterministic and small enough to run on CPU.
- Document shape, dtype, masking, and padding behavior explicitly.
- Avoid unrelated formatting or refactoring in a focused pull request.

Changes to attention masks should include a future-mutation test: altering a
key that is unavailable to a query must not change that query's output.
Changes involving padding should prove that invalid tokens cannot affect valid
outputs.

Changes to the reconstruction utilities should test deterministic masks,
valid-token selection, masked-only float32 MSE, equal stream averaging, and
gradient flow into a single head reused across streams. If mask seeding
changes, document whether batch reordering changes the result and whether the
change affects reproducibility across C++ standard-library implementations.

## Safe fixtures

Only generated or clearly redistributable fixtures belong in this repository.
Do not submit credentials, private endpoints, local absolute paths, external
datasets, model weights, cached features, run logs, or results derived from
data that cannot be redistributed.

## Pull requests

A pull request should explain the observable change, why it is needed, and how
it was tested. Keep commits reviewable. All tests and both synthetic
executables must pass before review.

By submitting a contribution, you agree that it is licensed under the Apache
License, Version 2.0, used by this project.
