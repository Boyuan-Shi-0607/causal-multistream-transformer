# Causal Multistream Transformer

Causal Multistream Transformer is a small C++17/LibTorch library for encoding
multiple asynchronous sequence streams without allowing a token to attend to
information that was not available at its decision time.

The encoder has three main parts:

- one shared context encoder applied independently to one or more context
  streams;
- a target encoder with causal self-attention;
- causal cross-attention from the target stream to the concatenated context
  memories.

Rotary position embeddings (RoPE) describe order in a caller-defined common
coordinate system. A separate `available_at` tensor controls causal access.
Keeping those concepts separate lets a token retain its natural position while
declaring when all information used to construct it became available.

This repository contains an initial implementation, synthetic examples, and
CPU tests. It has not been validated on GPU and makes no performance claims.

## Requirements

- A C++17 compiler
- CMake 3.22 or newer
- Python 3 with PyTorch/LibTorch available to CMake
- PyTorch 2.9.1 for the currently validated build

PyTorch/LibTorch is a separately installed third-party dependency and remains
governed by its own license.

## Build and test

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/cmt_synthetic
```

The synthetic executable performs a forward and backward pass on generated
signals. It is a gradient and integration check, not a training recipe or a
benchmark for a real task.

## Public API

The main types live in namespace `cmt`:

```cpp
cmt::EncoderConfig config;
config.input_dim = 8;
config.model_dim = 64;

cmt::CausalMultiStreamEncoder encoder(config);
cmt::EncoderOutput output = encoder->forward(target, contexts);
```

An `EncoderConfig` controls the input and model dimensions, attention heads,
context and target depth, feed-forward width, embeddings, dropout, and RoPE
base. A `StreamBatch` contains:

- `features`: `[B, L, F]`
- `positions`: `[B, L]`
- `available_at`: `[B, L]`
- `valid`: boolean `[B, L]`
- optional `segment_ids`: int32 or int64 `[B, L]`
- scalar `stream_id` and `scale_id`

Sequence lengths may differ between streams, but every stream must have a
positive length and the target must receive at least one context stream.
`available_at` must be int32 or int64. `positions` may be int32, int64,
float32, or float64; their dtype must be consistent across all streams.

See [the architecture notes](docs/architecture.md) and
[the data contract](docs/data_contract.md) before adapting the encoder to a
new source of sequences.

## Causality boundary

The encoder enforces attention masks from `available_at` metadata: a query may
read a valid key only when the key was available no later than the query.
Equality is allowed.

This does not make arbitrary inputs causal. Callers remain responsible for
constructing features and metadata using only information available at that
time. For example, a centered transform, a statistic computed over the full
sequence, or normalization fitted on future samples leaks information before
the encoder sees it.

## Scope

The project is an encoder library. It does not distribute datasets, model
weights, task-specific prediction heads, or a production training system.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and focused changes
to correctness, portability, documentation, and synthetic tests are welcome.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.
