# Architecture

## Overview

`cmt::CausalMultiStreamEncoder` consumes one target stream and one or more
context streams. Each stream carries both content and two notions of time:

- `positions` supplies the RoPE coordinate used to represent order and
  distance;
- `available_at` supplies the clock used to decide whether attention is
  permitted.

These tensors use caller-defined units, but all streams in one forward pass
must use the same unit and origin.

```text
context 0 ── shared context encoder ── memory 0 ─┐
context 1 ── shared context encoder ── memory 1 ─┼─ concatenate
      ...                                       ─┘       │
                                                        ▼
target ── causal target self-attention ── causal cross-attention
                                                        │
                                                        ▼
                                                 target memory
```

The same context encoder parameters are reused for every context stream. Each
context is encoded independently before its memory is concatenated with the
others. This preserves one shared representation function without allowing
context streams to exchange information during context encoding.

The target stack performs causal self-attention and then cross-attends to the
combined context memory. The number of context and target layers is configured
independently.

## Embeddings

Each token begins with a projection of its feature vector. Learned stream,
scale, and segment embeddings identify its source. Segment IDs are optional;
when omitted, the implementation uses the default segment.

RoPE is applied inside attention using `positions`. Every stream must use a
consistent coordinate system and position dtype. Positions are converted to
float32 while the phase, sine, and cosine factors are calculated, then the
sine and cosine factors are converted to the attention feature dtype. Their
absolute value is limited to `1,000,000`; subtract a common origin when a
source coordinate would exceed that range. RoPE changes how attention
represents distance; it does not decide which keys are visible.

## Causal attention

Let `a_q` be the `available_at` value for a valid query and `a_k` the value for
a valid key. The key is visible exactly when

\[
  a_k \leq a_q.
\]

Equality is intentionally allowed. A token completed at time `t` may be used
by another token whose decision point is also `t`.

The same rule applies to target self-attention, context self-attention, and
target-to-context cross-attention. Invalid keys are never visible. Invalid
padding is zeroed before it enters the attention stack and remains excluded by
the mask.

For an aggregated token, `available_at` must be the completion time of the
last source observation needed to compute that token. Its `position` may still
represent a start, center, end, or other domain-specific coordinate as long as
the convention is consistent.

## Configuration

The default `EncoderConfig` is:

| Field | Default | Meaning |
| --- | ---: | --- |
| `input_dim` | 8 | Feature width per token |
| `model_dim` | 64 | Hidden width |
| `heads` | 4 | Attention heads |
| `context_layers` | 2 | Shared context encoder depth |
| `target_layers` | 2 | Target self/cross-attention depth |
| `ffn_multiplier` | 4 | Feed-forward expansion |
| `stream_count` | 3 | Stream embedding vocabulary |
| `scale_count` | 3 | Scale embedding vocabulary |
| `segment_count` | 1 | Segment embedding vocabulary |
| `dropout` | 0 | Dropout probability |
| `rope_base` | 10000 | RoPE frequency base |

The model dimension must be divisible by the number of heads. Stream, scale,
and segment IDs must fall within their configured embedding vocabularies.

## Outputs

`EncoderOutput` contains:

- `target_memory`: encoded target tokens, `[B, L_target, model_dim]`;
- `context_memories`: one encoded tensor per context stream;
- `context_memory`: all context memories concatenated along the sequence axis.

The concatenation order matches the input context vector.

## Optional reconstruction warm-up

The encoder memories can be paired with the utilities in
`<cmt/pretraining.hpp>`. Mask construction and feature masking happen before
the encoder forward pass. The original, unmasked features remain the
reconstruction targets.

One `ReconstructionHead` is reused for all encoded streams. It contains a
shared normalization and hidden projection followed by one reconstruction
projection. Reusing the same module instance makes every stream update the
same head parameters.

For stream `s`, prediction `P_s`, original target `X_s`, boolean mask `M_s`,
and caller-selected feature set `J`, the loss is

\[
  L_s =
  \frac{
    \sum_{b,t,j \in J} M_{s,b,t}
      \left(P_{s,b,t,j} - X_{s,b,t,j}\right)^2
  }{
    \max\left(1, |J| \sum_{b,t} M_{s,b,t}\right)
  }.
\]

Squared error and reduction are accumulated in float32. For a non-empty set of
stream losses, `mean_stream_loss` returns their arithmetic mean, so each stream
has equal weight regardless of sequence length or number of masked tokens.

All masking choices and feature indices belong to the caller. The library does
not assign semantic meaning to a feature column. See
[pretraining.md](pretraining.md) for API details.

## What the mask does not guarantee

The mask only reasons about the metadata and tensors supplied to the model. It
cannot detect information already embedded in a feature. Leakage can still be
introduced by preprocessing, target construction, sample selection, or
normalization across future observations. See the
[data contract](data_contract.md) for the caller's responsibilities.
