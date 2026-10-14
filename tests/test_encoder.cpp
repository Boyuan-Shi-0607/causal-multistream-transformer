#include <cmt/encoder.hpp>

#include "test_support.hpp"

#include <torch/serialize.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

cmt::EncoderConfig small_config() {
  cmt::EncoderConfig config;
  config.input_dim = 8;
  config.model_dim = 16;
  config.heads = 4;
  config.context_layers = 2;
  config.target_layers = 2;
  config.ffn_multiplier = 2;
  config.stream_count = 3;
  config.scale_count = 3;
  config.segment_count = 1;
  config.dropout = 0.0;
  config.rope_base = 10000.0;
  return config;
}

cmt::StreamBatch stream(
    int64_t length,
    int64_t input_dim,
    int64_t stream_id,
    int64_t scale_id,
    const std::vector<int64_t>& positions,
    const std::vector<int64_t>& available_at) {
  cmt_test::require(
      static_cast<int64_t>(positions.size()) == length &&
          static_cast<int64_t>(available_at.size()) == length,
      "test stream metadata length mismatch");
  cmt::StreamBatch batch;
  batch.features = torch::randn({1, length, input_dim});
  batch.positions = torch::tensor(
      positions, torch::TensorOptions().dtype(torch::kInt64)).view({1, length});
  batch.available_at = torch::tensor(
      available_at, torch::TensorOptions().dtype(torch::kInt64)).view({1, length});
  batch.valid = torch::ones({1, length}, torch::kBool);
  batch.stream_id = stream_id;
  batch.scale_id = scale_id;
  return batch;
}

cmt::StreamBatch clone_stream(const cmt::StreamBatch& source) {
  cmt::StreamBatch copy;
  copy.features = source.features.clone();
  copy.positions = source.positions.clone();
  copy.available_at = source.available_at.clone();
  copy.valid = source.valid.clone();
  if (source.segment_ids.defined()) {
    copy.segment_ids = source.segment_ids.clone();
  }
  copy.stream_id = source.stream_id;
  copy.scale_id = source.scale_id;
  return copy;
}

struct Inputs {
  cmt::StreamBatch target;
  std::vector<cmt::StreamBatch> contexts;
};

Inputs valid_inputs(const cmt::EncoderConfig& config) {
  Inputs value;
  value.target = stream(
      4, config.input_dim, 0, 0,
      {0, 1, 2, 3},
      {3, 7, 11, 15});
  value.contexts.push_back(stream(
      3, config.input_dim, 1, 1,
      {-5, 0, 5},
      {1, 6, 20}));
  value.contexts.push_back(stream(
      5, config.input_dim, 2, 2,
      {-9, -4, 1, 6, 11},
      {2, 4, 9, 30, 31}));
  return value;
}

Inputs clone_inputs(const Inputs& source) {
  Inputs copy;
  copy.target = clone_stream(source.target);
  for (const auto& context : source.contexts) {
    copy.contexts.push_back(clone_stream(context));
  }
  return copy;
}

void require_output_finite(const cmt::EncoderOutput& output) {
  cmt_test::require_finite(output.target_memory, "target memory must be finite");
  cmt_test::require_finite(output.context_memory, "combined context memory must be finite");
  cmt_test::require(output.context_memories.size() == 2,
                    "one context memory is required for each context stream");
  for (const auto& memory : output.context_memories) {
    cmt_test::require_finite(memory, "per-stream context memory must be finite");
  }
}

void future_mutation_cannot_change_past_or_receive_gradient() {
  torch::manual_seed(401);
  const auto config = small_config();
  cmt::CausalMultiStreamEncoder encoder(config);
  encoder->eval();

  const Inputs baseline_inputs = valid_inputs(config);
  Inputs mutated_inputs = clone_inputs(baseline_inputs);

  mutated_inputs.target.features.narrow(1, 2, 2).mul_(17.0).add_(23.0);
  mutated_inputs.target.positions.narrow(1, 2, 2).add_(1000);
  mutated_inputs.target.available_at.narrow(1, 2, 2).add_(1000);

  mutated_inputs.contexts[0].features.narrow(1, 2, 1).mul_(-19.0).add_(31.0);
  mutated_inputs.contexts[0].positions.narrow(1, 2, 1).sub_(700);
  mutated_inputs.contexts[0].available_at.narrow(1, 2, 1).add_(1000);

  mutated_inputs.contexts[1].features.narrow(1, 2, 3).mul_(13.0).sub_(29.0);
  mutated_inputs.contexts[1].positions.narrow(1, 2, 3).add_(900);
  mutated_inputs.contexts[1].available_at.narrow(1, 2, 3).add_(1000);

  const auto baseline = encoder->forward(
      baseline_inputs.target, baseline_inputs.contexts);
  const auto mutated = encoder->forward(
      mutated_inputs.target, mutated_inputs.contexts);
  cmt_test::require_allclose(
      baseline.target_memory.narrow(1, 0, 2),
      mutated.target_memory.narrow(1, 0, 2),
      1e-6,
      1e-6,
      "future feature or metadata mutation changed a past target representation");

  Inputs gradient_inputs = valid_inputs(config);
  gradient_inputs.target.features.set_requires_grad(true);
  for (auto& context : gradient_inputs.contexts) {
    context.features.set_requires_grad(true);
  }
  const auto gradient_output = encoder->forward(
      gradient_inputs.target, gradient_inputs.contexts);
  gradient_output.target_memory
      .narrow(1, 0, 2)
      .select(2, 0)
      .square()
      .sum()
      .backward();

  const double visible_target_gradient = gradient_inputs.target.features.grad()
      .narrow(1, 0, 2)
      .abs()
      .max()
      .item<double>();
  const double visible_context_gradient = std::max(
      gradient_inputs.contexts[0].features.grad().narrow(1, 0, 2)
          .abs().max().item<double>(),
      gradient_inputs.contexts[1].features.grad().narrow(1, 0, 2)
          .abs().max().item<double>());
  cmt_test::require(
      visible_target_gradient > 1e-10,
      "past target inputs must receive a nonzero gradient from past outputs");
  cmt_test::require(
      visible_context_gradient > 1e-10,
      "visible context inputs must receive a nonzero gradient from past outputs");

  cmt_test::require_zero(
      gradient_inputs.target.features.grad().narrow(1, 2, 2),
      0.0,
      "future target features received gradients from past outputs");
  cmt_test::require_zero(
      gradient_inputs.contexts[0].features.grad().narrow(1, 2, 1),
      0.0,
      "future context features received gradients from past outputs");
  cmt_test::require_zero(
      gradient_inputs.contexts[1].features.grad().narrow(1, 2, 3),
      0.0,
      "future context suffix received gradients from past outputs");
}

void invalid_nan_padding_is_isolated() {
  torch::manual_seed(402);
  const auto config = small_config();
  cmt::CausalMultiStreamEncoder encoder(config);
  encoder->eval();

  Inputs clean = valid_inputs(config);
  clean.target.positions = clean.target.positions.to(torch::kFloat64);
  clean.target.segment_ids = torch::zeros_like(
      clean.target.valid, torch::TensorOptions().dtype(torch::kInt64));
  for (auto& context : clean.contexts) {
    context.positions = context.positions.to(torch::kFloat64);
    context.segment_ids = torch::zeros_like(
        context.valid, torch::TensorOptions().dtype(torch::kInt64));
  }
  clean.target.valid[0][3].fill_(false);
  clean.contexts[0].valid[0][2].fill_(false);
  clean.contexts[1].valid[0][1].fill_(false);
  clean.target.features[0][3].zero_();
  clean.contexts[0].features[0][2].zero_();
  clean.contexts[1].features[0][1].zero_();

  Inputs padded = clone_inputs(clean);
  padded.target.features[0][3].fill_(
      std::numeric_limits<float>::quiet_NaN());
  padded.contexts[0].features[0][2].fill_(
      std::numeric_limits<float>::quiet_NaN());
  padded.contexts[1].features[0][1].fill_(
      std::numeric_limits<float>::quiet_NaN());
  padded.target.positions[0][3].fill_(
      std::numeric_limits<double>::quiet_NaN());
  padded.contexts[0].positions[0][2].fill_(
      std::numeric_limits<double>::quiet_NaN());
  padded.contexts[1].positions[0][1].fill_(
      std::numeric_limits<double>::quiet_NaN());
  padded.target.available_at[0][3].fill_(
      std::numeric_limits<int64_t>::max());
  padded.contexts[0].available_at[0][2].fill_(
      std::numeric_limits<int64_t>::max());
  padded.contexts[1].available_at[0][1].fill_(
      std::numeric_limits<int64_t>::max());
  padded.target.segment_ids[0][3].fill_(
      std::numeric_limits<int64_t>::max());
  padded.contexts[0].segment_ids[0][2].fill_(
      std::numeric_limits<int64_t>::max());
  padded.contexts[1].segment_ids[0][1].fill_(
      std::numeric_limits<int64_t>::max());

  padded.target.features.set_requires_grad(true);
  for (auto& context : padded.contexts) {
    context.features.set_requires_grad(true);
  }

  const auto clean_output = encoder->forward(clean.target, clean.contexts);
  const auto padded_output = encoder->forward(padded.target, padded.contexts);
  require_output_finite(padded_output);
  cmt_test::require_allclose(
      clean_output.target_memory.narrow(1, 0, 3),
      padded_output.target_memory.narrow(1, 0, 3),
      0.0,
      0.0,
      "NaN padding changed valid target representations");
  cmt_test::require_zero(
      padded_output.target_memory.select(1, 3),
      0.0,
      "an invalid target token must have zero representation");
  cmt_test::require_zero(
      padded_output.context_memories[0].select(1, 2),
      0.0,
      "an invalid context token must have zero representation");
  cmt_test::require_zero(
      padded_output.context_memories[1].select(1, 1),
      0.0,
      "NaN context padding must not survive encoding");

  padded_output.target_memory
      .narrow(1, 0, 3)
      .select(2, 0)
      .square()
      .sum()
      .backward();
  cmt_test::require_finite(
      padded.target.features.grad(),
      "target feature gradients must remain finite with NaN padding");
  cmt_test::require_finite(
      padded.contexts[0].features.grad(),
      "first context gradients must remain finite with NaN padding");
  cmt_test::require_finite(
      padded.contexts[1].features.grad(),
      "second context gradients must remain finite with NaN padding");
  cmt_test::require_zero(
      padded.target.features.grad().select(1, 3),
      0.0,
      "invalid target padding must receive zero gradient");
  cmt_test::require_zero(
      padded.contexts[0].features.grad().select(1, 2),
      0.0,
      "invalid first-context padding must receive zero gradient");
  cmt_test::require_zero(
      padded.contexts[1].features.grad().select(1, 1),
      0.0,
      "invalid second-context padding must receive zero gradient");
}

void malformed_config_and_batches_fail_fast() {
  auto config = small_config();
  auto invalid_width = config;
  invalid_width.model_dim = 15;
  cmt_test::require_throws(
      [&] { cmt::CausalMultiStreamEncoder model(invalid_width); },
      "model width not divisible by heads must be rejected");

  auto invalid_stream_count = config;
  invalid_stream_count.stream_count = 0;
  cmt_test::require_throws(
      [&] { cmt::CausalMultiStreamEncoder model(invalid_stream_count); },
      "zero stream count must be rejected");

  cmt::CausalMultiStreamEncoder encoder(config);
  encoder->eval();
  Inputs inputs = valid_inputs(config);
  cmt_test::require_throws(
      [&] { encoder->forward(inputs.target, {}); },
      "at least one context stream is required");

  Inputs wrong_shape = clone_inputs(inputs);
  wrong_shape.contexts[0].positions = torch::zeros({1, 2}, torch::kInt64);
  cmt_test::require_throws(
      [&] { encoder->forward(wrong_shape.target, wrong_shape.contexts); },
      "metadata length mismatch must be rejected");

  Inputs bad_identifier = clone_inputs(inputs);
  bad_identifier.contexts[0].stream_id = config.stream_count;
  cmt_test::require_throws(
      [&] { encoder->forward(bad_identifier.target, bad_identifier.contexts); },
      "out-of-range stream identifier must be rejected");

  Inputs nonfinite = clone_inputs(inputs);
  nonfinite.contexts[0].features[0][0][0].fill_(
      std::numeric_limits<float>::quiet_NaN());
  cmt_test::require_throws(
      [&] { encoder->forward(nonfinite.target, nonfinite.contexts); },
      "non-finite valid features must be rejected");

  Inputs half_positions = clone_inputs(inputs);
  half_positions.target.positions =
      half_positions.target.positions.to(torch::kFloat16);
  cmt_test::require_throws(
      [&] { encoder->forward(half_positions.target, half_positions.contexts); },
      "float16 positions must be rejected by the encoder");
}

void serialization_round_trip_preserves_outputs() {
  torch::manual_seed(403);
  const auto config = small_config();
  cmt::CausalMultiStreamEncoder original(config);
  original->eval();
  const Inputs inputs = valid_inputs(config);
  const auto expected = original->forward(inputs.target, inputs.contexts);

  torch::serialize::OutputArchive output_archive;
  original->save(output_archive);
  std::stringstream buffer(
      std::ios::in | std::ios::out | std::ios::binary);
  output_archive.save_to(buffer);
  buffer.seekg(0);

  cmt::CausalMultiStreamEncoder restored(config);
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(buffer);
  restored->load(input_archive);
  restored->eval();
  const auto actual = restored->forward(inputs.target, inputs.contexts);

  cmt_test::require_allclose(
      expected.target_memory, actual.target_memory, 0.0, 0.0,
      "serialized target representations changed after reload");
  cmt_test::require_allclose(
      expected.context_memory, actual.context_memory, 0.0, 0.0,
      "serialized context representations changed after reload");
  cmt_test::require(actual.context_memories.size() == expected.context_memories.size(),
                    "serialization changed the context stream count");
  for (size_t index = 0; index < expected.context_memories.size(); ++index) {
    cmt_test::require_allclose(
        expected.context_memories[index], actual.context_memories[index], 0.0, 0.0,
        "serialized per-stream context representation changed after reload");
  }
}

}  // namespace

int main() {
  int failures = 0;
  cmt_test::run_case(
      "future mutation and gradients are causal",
      future_mutation_cannot_change_past_or_receive_gradient,
      failures);
  cmt_test::run_case(
      "invalid NaN padding is isolated",
      invalid_nan_padding_is_isolated,
      failures);
  cmt_test::run_case(
      "malformed config and batches fail fast",
      malformed_config_and_batches_fail_fast,
      failures);
  cmt_test::run_case(
      "serialization round trip",
      serialization_round_trip_preserves_outputs,
      failures);
  return failures == 0 ? 0 : 1;
}
