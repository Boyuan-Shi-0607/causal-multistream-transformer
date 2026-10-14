#include <cmt/encoder.hpp>
#include <cmt/pretraining.hpp>

#include "test_support.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

double finite_gradient_energy(const torch::nn::Module& module) {
  double energy = 0.0;
  for (const auto& parameter : module.parameters()) {
    if (!parameter.grad().defined()) {
      continue;
    }
    cmt_test::require_finite(
        parameter.grad(), "module parameter gradient must be finite");
    energy += parameter.grad().detach().to(torch::kFloat64)
                  .square().sum().item<double>();
  }
  return energy;
}

cmt::StreamBatch make_stream(
    int64_t length,
    int64_t feature_dim,
    int64_t stream_id,
    int64_t scale_id,
    int64_t stride) {
  cmt::StreamBatch stream;
  stream.features = torch::randn({1, length, feature_dim});
  stream.features.select(-1, feature_dim - 1).zero_();
  stream.positions = (torch::arange(length, torch::kInt64) * stride)
                         .view({1, length});
  stream.available_at = stream.positions.clone();
  stream.valid = torch::ones({1, length}, torch::kBool);
  stream.stream_id = stream_id;
  stream.scale_id = scale_id;
  return stream;
}

void selected_masked_mse_is_exact_and_sparse() {
  auto prediction = torch::tensor(
      {{{1.0, 40.0, 3.0, -20.0}, {7.0, 8.0, 9.0, 10.0}}},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto target = torch::zeros({1, 2, 4}, torch::kFloat64);
  const auto mask = torch::tensor(
      {{true, false}}, torch::TensorOptions().dtype(torch::kBool));

  const auto loss = cmt::masked_reconstruction_loss(
      prediction, target, mask, {0, 2});
  cmt_test::require(
      loss.scalar_type() == torch::kFloat32,
      "masked reconstruction loss must accumulate in float32");
  cmt_test::require_allclose(
      loss, torch::tensor(5.0f), 0.0, 0.0,
      "selected masked MSE disagrees with the hand calculation");

  loss.backward();
  const auto gradient = prediction.grad();
  cmt_test::require_finite(gradient, "prediction gradient must be finite");
  cmt_test::require_allclose(
      gradient[0][0][0], torch::tensor(1.0, torch::kFloat64), 0.0, 0.0,
      "first selected gradient is incorrect");
  cmt_test::require_allclose(
      gradient[0][0][2], torch::tensor(3.0, torch::kFloat64), 0.0, 0.0,
      "second selected gradient is incorrect");
  cmt_test::require_zero(
      gradient[0][0].index_select(
          0, torch::tensor({1, 3}, torch::kInt64)),
      0.0,
      "unselected features must receive zero gradient");
  cmt_test::require_zero(
      gradient.select(1, 1), 0.0,
      "unmasked tokens must receive zero gradient");
}

void empty_masks_remain_differentiable_zero() {
  auto prediction = torch::randn({2, 3, 4}).set_requires_grad(true);
  const auto target = torch::randn({2, 3, 4});
  const auto mask = torch::zeros({2, 3}, torch::kBool);

  const auto loss = cmt::masked_reconstruction_loss(
      prediction, target, mask, {0, 2});
  cmt_test::require(
      loss.requires_grad(), "empty-mask loss must retain an autograd path");
  cmt_test::require_zero(loss, 0.0, "empty-mask loss must be exactly zero");
  loss.backward();
  cmt_test::require_zero(
      prediction.grad(), 0.0,
      "empty-mask loss must produce zero prediction gradients");
}

void nonfinite_values_are_checked_only_when_selected_and_masked() {
  auto prediction = torch::zeros({1, 2, 3});
  auto target = torch::zeros({1, 2, 3});
  const auto mask = torch::tensor(
      {{true, false}}, torch::TensorOptions().dtype(torch::kBool));
  prediction[0][0][0].fill_(2.0);
  prediction[0][0][2].fill_(4.0);
  prediction[0][0][1].fill_(std::numeric_limits<float>::quiet_NaN());
  prediction[0][1].fill_(std::numeric_limits<float>::quiet_NaN());
  target[0][1].fill_(std::numeric_limits<float>::quiet_NaN());

  const auto finite_loss = cmt::masked_reconstruction_loss(
      prediction, target, mask, {0, 2});
  cmt_test::require_allclose(
      finite_loss, torch::tensor(10.0f), 0.0, 0.0,
      "unselected or unmasked non-finite values polluted the loss");

  auto invalid_prediction = prediction.clone();
  invalid_prediction[0][0][0].fill_(
      std::numeric_limits<float>::quiet_NaN());
  cmt_test::require_throws(
      [&] {
        cmt::masked_reconstruction_loss(
            invalid_prediction, target, mask, {0, 2});
      },
      "a non-finite selected masked prediction must be rejected");

  auto invalid_target = target.clone();
  invalid_target[0][0][2].fill_(
      std::numeric_limits<float>::quiet_NaN());
  cmt_test::require_throws(
      [&] {
        cmt::masked_reconstruction_loss(
            prediction, invalid_target, mask, {0, 2});
      },
      "a non-finite selected masked target must be rejected");
}

void stream_losses_are_reduced_with_equal_weight() {
  auto first = torch::tensor(1.0, torch::TensorOptions().requires_grad(true));
  auto second = torch::tensor(4.0, torch::TensorOptions().requires_grad(true));
  auto third = torch::tensor(9.0, torch::TensorOptions().requires_grad(true));
  const auto mean = cmt::mean_stream_loss({first, second, third});
  const auto expected = torch::tensor(14.0 / 3.0);
  cmt_test::require_allclose(
      mean, expected, 1e-7, 1e-7,
      "stream losses must be reduced by an equal arithmetic mean");

  const auto pooled =
      (first * 1.0 + second * 10.0 + third * 100.0) / 111.0;
  cmt_test::require(
      (mean - pooled).abs().item<double>() > 1.0,
      "equal stream reduction must differ from element-count pooling");

  mean.backward();
  const auto one_third = torch::tensor(1.0 / 3.0);
  cmt_test::require_allclose(first.grad(), one_third, 1e-7, 1e-7,
                             "first stream gradient is not one third");
  cmt_test::require_allclose(second.grad(), one_third, 1e-7, 1e-7,
                             "second stream gradient is not one third");
  cmt_test::require_allclose(third.grad(), one_third, 1e-7, 1e-7,
                             "third stream gradient is not one third");

  cmt_test::require_throws(
      [] { cmt::mean_stream_loss({}); },
      "an empty stream-loss collection must be rejected");
}

void span_masks_are_deterministic_and_respect_validity() {
  cmt::SpanMaskConfig config;
  config.probabilities = {0.35, 0.55};
  config.min_span = 1;
  config.max_span = 3;

  const double probability = cmt::sample_mask_probability(config, 8123, 17);
  cmt_test::require(
      probability == 0.35 || probability == 0.55,
      "sampled probability must come from the configured list");
  cmt_test::require(
      probability == cmt::sample_mask_probability(config, 8123, 17),
      "probability sampling must be deterministic");
  bool found_different_probability = false;
  for (int64_t step = 18; step < 82; ++step) {
    found_different_probability = found_different_probability ||
        cmt::sample_mask_probability(config, 8123, step) != probability;
  }
  cmt_test::require(
      found_different_probability,
      "the configured probability schedule did not vary across many steps");

  const auto valid = torch::tensor(
      {{true, true, false, true, true, false, true, true, true, false, true, true},
       {false, true, true, false, true, false, true, true, false, true, false, true},
       {false, false, false, false, false, false, false, false, false, false, false, false}},
      torch::TensorOptions().dtype(torch::kBool));
  const auto sample_ids = torch::tensor(
      {11, 29, 47}, torch::TensorOptions().dtype(torch::kInt64));
  constexpr double mask_probability = 0.55;
  const auto first = cmt::make_span_mask(
      valid, sample_ids, mask_probability, 1, 3, 991, 23, 5);
  const auto repeat = cmt::make_span_mask(
      valid, sample_ids, mask_probability, 1, 3, 991, 23, 5);
  cmt_test::require(first.scalar_type() == torch::kBool,
                    "span mask must be boolean");
  cmt_test::require(first.equal(repeat),
                    "same span-mask inputs must reproduce bitwise");
  cmt_test::require(
      !first.logical_and(valid.logical_not()).any().item<bool>(),
      "invalid tokens must never be masked");

  for (int64_t row = 0; row < valid.size(0); ++row) {
    const int64_t valid_count = valid[row].sum().item<int64_t>();
    const int64_t expected = valid_count == 0
        ? 0
        : std::max<int64_t>(
              1,
              static_cast<int64_t>(
                  std::llround(valid_count * mask_probability)));
    cmt_test::require(
        first[row].sum().item<int64_t>() == expected,
        "span mask did not cover the requested number of valid tokens");
  }
  cmt_test::require(!first[2].any().item<bool>(),
                    "an all-invalid row must remain fully unmasked");

  bool step_changes_mask = false;
  bool key_changes_mask = false;
  for (uint64_t offset = 1; offset <= 64; ++offset) {
    step_changes_mask = step_changes_mask || !first.equal(cmt::make_span_mask(
        valid, sample_ids, mask_probability, 1, 3, 991, 23 + offset, 5));
    key_changes_mask = key_changes_mask || !first.equal(cmt::make_span_mask(
        valid, sample_ids, mask_probability, 1, 3, 991, 23, 5 + offset));
  }
  cmt_test::require(step_changes_mask,
                    "changing the step never changed a span mask");
  cmt_test::require(key_changes_mask,
                    "changing the stream key never changed a span mask");
}

void feature_masking_clones_and_marks_inputs() {
  cmt::StreamBatch input;
  input.features = torch::arange(20, torch::kFloat32).view({1, 4, 5});
  input.features.select(-1, 4).fill_(7.0);
  input.positions = torch::arange(4, torch::kInt64).view({1, 4});
  input.available_at = input.positions.clone();
  input.valid = torch::tensor(
      {{true, true, true, false}}, torch::TensorOptions().dtype(torch::kBool));
  input.stream_id = 1;
  input.scale_id = 2;
  const auto original = input.features.clone();

  cmt::SpanMaskConfig config;
  config.probabilities = {0.35, 0.55};
  config.min_span = 1;
  config.max_span = 3;
  config.masked_feature_indices = {1, 3};
  config.indicator_feature_index = 4;
  const auto mask = torch::tensor(
      {{false, true, true, false}}, torch::TensorOptions().dtype(torch::kBool));
  const auto masked = cmt::mask_span_features(input, mask, config);

  cmt_test::require(input.features.equal(original),
                    "feature masking modified its input in place");
  cmt_test::require(masked.features.data_ptr() != input.features.data_ptr(),
                    "feature masking must return cloned feature storage");
  cmt_test::require_zero(
      masked.features.select(-1, 1).masked_select(mask), 0.0,
      "first selected feature was not cleared on the mask");
  cmt_test::require_zero(
      masked.features.select(-1, 3).masked_select(mask), 0.0,
      "second selected feature was not cleared on the mask");
  cmt_test::require(
      masked.features.select(-1, 4).to(torch::kBool).equal(mask),
      "indicator feature must equal the span mask");
  cmt_test::require(
      masked.features.select(-1, 0).equal(original.select(-1, 0)) &&
          masked.features.select(-1, 2).equal(original.select(-1, 2)),
      "unselected features changed during masking");
  cmt_test::require(
      masked.features.select(-1, 1).masked_select(mask.logical_not()).equal(
          original.select(-1, 1).masked_select(mask.logical_not())),
      "selected features changed outside the mask");
  cmt_test::require(
      masked.positions.equal(input.positions) &&
          masked.available_at.equal(input.available_at) &&
          masked.valid.equal(input.valid) &&
          masked.stream_id == input.stream_id &&
          masked.scale_id == input.scale_id,
      "feature masking changed stream metadata");

  auto invalid_mask = mask.clone();
  invalid_mask[0][3].fill_(true);
  cmt_test::require_throws(
      [&] { cmt::mask_span_features(input, invalid_mask, config); },
      "masking an invalid token must be rejected");
}

std::vector<torch::Tensor> capture_head_gradients(
    cmt::ReconstructionHead& head,
    const torch::Tensor& memory,
    const torch::Tensor& target,
    const torch::Tensor& mask) {
  head->zero_grad();
  const auto prediction = head->forward(memory);
  const auto loss = cmt::masked_reconstruction_loss(
      prediction, target, mask, {0, 2});
  loss.backward();
  std::vector<torch::Tensor> gradients;
  for (const auto& parameter : head->parameters()) {
    cmt_test::require(parameter.grad().defined(),
                      "head parameter gradient must be defined");
    gradients.push_back(parameter.grad().detach().clone());
  }
  return gradients;
}

void one_head_accumulates_equal_multistream_gradients() {
  torch::manual_seed(731);
  cmt::ReconstructionHead head(8, 4, 0.0);
  head->eval();
  const auto memory_a = torch::randn({1, 2, 8});
  const auto memory_b = torch::randn({1, 3, 8});
  const auto target_a = torch::randn({1, 2, 4});
  const auto target_b = torch::randn({1, 3, 4});
  const auto mask_a = torch::ones({1, 2}, torch::kBool);
  const auto mask_b = torch::ones({1, 3}, torch::kBool);

  const auto gradients_a = capture_head_gradients(
      head, memory_a, target_a, mask_a);
  const auto gradients_b = capture_head_gradients(
      head, memory_b, target_b, mask_b);

  head->zero_grad();
  const auto loss_a = cmt::masked_reconstruction_loss(
      head->forward(memory_a), target_a, mask_a, {0, 2});
  const auto loss_b = cmt::masked_reconstruction_loss(
      head->forward(memory_b), target_b, mask_b, {0, 2});
  cmt::mean_stream_loss({loss_a, loss_b}).backward();
  const auto combined = head->parameters();
  cmt_test::require(combined.size() == gradients_a.size() &&
                        combined.size() == gradients_b.size(),
                    "head parameter count changed between passes");
  for (size_t index = 0; index < combined.size(); ++index) {
    cmt_test::require_allclose(
        combined[index].grad(),
        (gradients_a[index] + gradients_b[index]) / 2.0,
        2e-5,
        2e-6,
        "shared-head gradient is not the equal mean of stream gradients");
  }
}

void encoder_and_head_receive_finite_nonzero_gradients() {
  torch::manual_seed(941);
  cmt::EncoderConfig encoder_config;
  encoder_config.input_dim = 5;
  encoder_config.model_dim = 16;
  encoder_config.heads = 4;
  encoder_config.context_layers = 1;
  encoder_config.target_layers = 1;
  encoder_config.ffn_multiplier = 2;
  encoder_config.stream_count = 2;
  encoder_config.scale_count = 2;
  encoder_config.segment_count = 1;
  encoder_config.dropout = 0.0;

  cmt::SpanMaskConfig mask_config;
  mask_config.probabilities = {0.35, 0.55};
  mask_config.min_span = 1;
  mask_config.max_span = 3;
  mask_config.masked_feature_indices = {0, 2};
  mask_config.indicator_feature_index = 4;

  cmt::CausalMultiStreamEncoder encoder(encoder_config);
  cmt::ReconstructionHead head(
      encoder_config.model_dim, encoder_config.input_dim, 0.0);
  encoder->train();
  head->train();

  const auto target = make_stream(6, encoder_config.input_dim, 0, 0, 1);
  const auto context = make_stream(4, encoder_config.input_dim, 1, 1, 2);
  const auto sample_ids = torch::tensor(
      {17}, torch::TensorOptions().dtype(torch::kInt64));
  const auto target_mask = cmt::make_span_mask(
      target.valid, sample_ids, 0.55, 1, 3, 119, 7, 0);
  const auto context_mask = cmt::make_span_mask(
      context.valid, sample_ids, 0.55, 1, 3, 119, 7, 1);
  const auto masked_target = cmt::mask_span_features(
      target, target_mask, mask_config);
  const auto masked_context = cmt::mask_span_features(
      context, context_mask, mask_config);

  const auto encoded = encoder->forward(masked_target, {masked_context});
  const auto target_loss = cmt::masked_reconstruction_loss(
      head->forward(encoded.target_memory),
      target.features,
      target_mask,
      {0, 2});
  const auto context_loss = cmt::masked_reconstruction_loss(
      head->forward(encoded.context_memories.at(0)),
      context.features,
      context_mask,
      {0, 2});
  const auto loss = cmt::mean_stream_loss({target_loss, context_loss});
  cmt_test::require_finite(loss, "integrated reconstruction loss must be finite");
  loss.backward();

  cmt_test::require(
      finite_gradient_energy(*encoder) > 0.0,
      "encoder must receive a nonzero finite reconstruction gradient");
  cmt_test::require(
      finite_gradient_energy(*head) > 0.0,
      "reconstruction head must receive a nonzero finite gradient");
}

}  // namespace

int main() {
  int failures = 0;
  cmt_test::run_case(
      "selected masked MSE is exact and sparse",
      selected_masked_mse_is_exact_and_sparse,
      failures);
  cmt_test::run_case(
      "empty masks remain differentiable zero",
      empty_masks_remain_differentiable_zero,
      failures);
  cmt_test::run_case(
      "non-finite values checked on selected mask",
      nonfinite_values_are_checked_only_when_selected_and_masked,
      failures);
  cmt_test::run_case(
      "stream losses use equal weight",
      stream_losses_are_reduced_with_equal_weight,
      failures);
  cmt_test::run_case(
      "span masks deterministic and validity-safe",
      span_masks_are_deterministic_and_respect_validity,
      failures);
  cmt_test::run_case(
      "feature masking clones and marks inputs",
      feature_masking_clones_and_marks_inputs,
      failures);
  cmt_test::run_case(
      "one head accumulates equal multistream gradients",
      one_head_accumulates_equal_multistream_gradients,
      failures);
  cmt_test::run_case(
      "encoder and head receive reconstruction gradients",
      encoder_and_head_receive_finite_nonzero_gradients,
      failures);
  return failures == 0 ? 0 : 1;
}
