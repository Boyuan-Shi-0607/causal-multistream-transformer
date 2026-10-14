# Data contract

The encoder accepts one target `StreamBatch` and a non-empty vector of context
`StreamBatch` values. The target and all contexts must describe the same batch
of examples.

## Tensor shapes

For a stream with batch size `B`, sequence length `L`, and configured input
width `F`:

| Field | Shape | Contract |
| --- | --- | --- |
| `features` | `[B, L, F]` | Floating-point token features |
| `positions` | `[B, L]` | RoPE coordinates; int32, int64, float32, or float64 |
| `available_at` | `[B, L]` | Causal availability coordinates; int32 or int64 |
| `valid` | `[B, L]` | Boolean token-validity mask |
| `segment_ids` | `[B, L]` | Optional int32 or int64 segment IDs |

Every sequence length must be positive. Context lengths may differ from each
other and from the target length. `features.size(2)` must equal
`EncoderConfig::input_dim`.

All tensors, encoder parameters, and buffers participating in a forward pass
must be on the same device. Feature tensors must share one dtype across all
streams and that dtype must match the encoder parameters. Batch dimensions
must agree across streams. Stream, scale, and segment IDs must be within the
ranges declared by the configuration.

## Metadata types

`valid` must be boolean. `segment_ids`, when provided, must be int32 or int64.
`available_at` must be int32 or int64, and every stream in a forward pass must
use the same availability dtype. `positions` may be int32, int64, float32, or
float64, and likewise must share one dtype across streams. Float16 and
bfloat16 metadata are rejected; do not downcast metadata even when feature
computation or model weights use reduced precision.

Valid position values must be finite and have absolute value no greater than
`1,000,000`. Choose a numeric unit that represents all streams without
ambiguous rounding, and use the same origin and unit for every stream in a
forward pass. Position values are converted to float64 for RoPE sine and cosine
calculation, then converted to the attention feature dtype.

## Position and availability are different

`positions` describes where a token belongs in the sequence. `available_at`
describes when the token may first be consumed.

For a token aggregated from several source observations, set `available_at` to
the completion time of the last observation used by the aggregation. Do not
use the start or center of the aggregation window merely because that value is
used as its display position.

The encoder permits a key when:

```text
key.valid && key.available_at <= query.available_at
```

Queries with the same availability time may see one another when otherwise
valid. If the application requires a stricter ordering for simultaneous
events, encode that ordering into the availability coordinate.

## Padding

Padding tokens must have `valid == false`. Their feature and metadata values
must not be treated as meaningful. The implementation zeros invalid token
representations and masks invalid keys, but callers should still initialize
padding consistently to simplify diagnostics.

## Caller responsibilities

The attention mask is one layer of a causal pipeline. Before calling the
encoder, the caller must ensure that every valid feature was computable at its
declared availability time.

Common sources of leakage include:

- centered or bidirectional transforms;
- statistics computed from a complete sequence and attached to early tokens;
- normalization fitted with future examples or future portions of one sample;
- aggregates whose availability is marked before their final input arrives;
- imputation that consults later observations;
- sample filtering based on outcomes that were unknown at decision time.

Fit reusable transforms on an appropriate training partition, and compute
online statistics with causal updates. Mutation tests are useful: changing an
unavailable future suffix should not change any earlier encoder output.

## Outputs and padding

The encoder returns a target memory, a combined context memory, and the
per-context memories. The per-context output order matches the input vector.
Invalid positions should not be interpreted as model predictions, even though
their stored representation is zeroed.

## Repository fixtures

Examples and tests in this repository use generated signals only. No external
dataset or pretrained weight is required by the data contract.
