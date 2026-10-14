# Masked reconstruction warm-up

The optional utilities in `<cmt/pretraining.hpp>` provide a small,
task-agnostic masked-reconstruction component for `CausalMultiStreamEncoder`.
They do not choose feature semantics or masking parameters.

## Configuration

`SpanMaskConfig` contains:

- `probabilities`: candidate mask probabilities;
- `min_span` and `max_span`: inclusive span-length bounds;
- `masked_feature_indices`: input columns cleared at masked tokens;
- `indicator_feature_index`: input column set at masked tokens.

The caller must populate these fields for its own tensor contract. The library
does not provide task-specific feature indices or recommended masking values.
Columns selected for reconstruction should be a subset of
`masked_feature_indices` and must not include `indicator_feature_index`.

`sample_mask_probability(config, seed, step)` reproducibly chooses one
candidate for a given seed, step, and C++ standard-library implementation.
Candidate probabilities must be valid for mask sampling.

## Mask generation

```cpp
torch::Tensor make_span_mask(
    const torch::Tensor& valid,
    const torch::Tensor& sample_ids,
    double probability,
    int64_t min_span,
    int64_t max_span,
    uint64_t seed,
    int64_t step,
    uint64_t stream_key);
```

`valid` is boolean `[B,L]`; `sample_ids` is int32 or int64 `[B]`. Both tensors
must be on CPU. The function returns a CPU boolean `[B,L]` mask and never
selects invalid padding. Probability must be finite and in `(0,1]`, step must
be non-negative, and span bounds must satisfy `0 < min_span <= max_span`. It
draws contiguous candidate spans and fills any remaining quota from unselected
valid tokens.

Mask generation is reproducible for the complete argument set, batch layout,
and C++ standard-library implementation. The random seed incorporates `seed`,
`step`, `sample_id`, batch row, and `stream_key`. The batch-row contribution
means that permuting rows can change the masks. Preserve row order when exact
mask replay is required. Although the random engine sequence is standardized,
standard-library distributions and shuffle are not guaranteed to map that
sequence identically across different implementations. Persist masks directly
when replay must cross toolchains.

Use a distinct stable `stream_key` when independently masking multiple streams.

## Feature masking

```cpp
cmt::StreamBatch mask_span_features(
    const cmt::StreamBatch& stream,
    const torch::Tensor& mask,
    const cmt::SpanMaskConfig& config);
```

The function returns a stream whose feature tensor is cloned. At selected
tokens it clears each `masked_feature_indices` column. It replaces the entire
`indicator_feature_index` column with the mask converted to the feature dtype,
so the column is one at selected tokens and zero elsewhere. The masked feature
list must be non-empty, unique, in range, and must not include the indicator
column. Features, validity, and mask must share one device. Since
`make_span_mask` returns a CPU mask, apply it before device transfer or move it
to the stream device. Positions, availability metadata, validity, segment IDs,
stream ID, and scale ID retain their input meaning.

Keep the original features separately. They are the targets for
reconstruction; the masked stream is the encoder input.

The masking API cannot infer semantic dependencies between columns. Any input
left visible, including a derived channel, can make a reconstruction target
directly recoverable. The caller is responsible for clearing every column that
would reveal its selected reconstruction targets.

## Shared reconstruction head

```cpp
cmt::ReconstructionHead head(model_dim, feature_dim, dropout);

const auto first_prediction = head->forward(first_memory);
const auto second_prediction = head->forward(second_memory);
```

Use one head instance for every encoded stream. The head applies a shared
normalization, hidden projection, activation, dropout, and reconstruction
projection. Reusing one instance ensures that gradients from all stream losses
update the same parameters.

Creating a separate head per stream is a different parameter-sharing contract.

## Masked MSE

```cpp
const auto loss = cmt::masked_reconstruction_loss(
    prediction,
    original_features,
    mask,
    reconstruction_feature_indices);
```

Prediction is `[B,L,F_prediction]`, target is `[B,L,F_target]`, and mask is
boolean `[B,L]`. Batch and length dimensions must match; the non-empty, unique
feature-index vector must be in range for both feature dimensions. The function
computes elementwise squared error only at selected token-column pairs.
Predictions and targets are converted to float32 before the error and
reduction, and the sum is normalized by the number of selected scalar
elements. An empty mask returns a differentiable zero.

Because masking and loss are separate functions, the loss cannot enforce the
relationship between their index vectors. Reconstruction indices should be a
subset of the columns cleared from the encoder input and must exclude the mask
indicator. This loss evaluates reconstruction at selected tokens; it is not a
metric for predicting later observations.

Combine a non-empty vector of per-stream losses with:

```cpp
const auto total = cmt::mean_stream_loss(stream_losses);
```

This is an arithmetic mean, so every stream contributes equal weight after its
own masked-element normalization.

## Synthetic learning check

`cmt_train_synthetic` builds generated multistream signals, runs the encoder
and one shared reconstruction head, and updates them with AdamW and gradient
clipping. It evaluates a fixed generated validation set before and after the
requested number of steps and reports the initial error, final error, and their
ratio.

```bash
./build/cmt_train_synthetic --steps 120
```

The only supported option is `--steps`; omitting it uses the built-in synthetic
check length. The executable reads no dataset and writes no model or cache
files. Its purpose is to verify that the public components can learn a small
generated reconstruction problem; its numbers are not a performance claim for
another task or data source.

## Scope of reproducibility

The APIs make the mask, corruption, parameter sharing, and loss reduction
explicit and testable. Reproducing a complete experiment also requires the
caller's feature construction, tensors, parameter choices, batching, optimizer
settings, and evaluation protocol. Those choices are outside this component.
