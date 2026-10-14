#include "cmt/pretraining.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace cmt {
namespace {

uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

bool is_integer_type(torch::ScalarType type) {
  return type == torch::kInt32 || type == torch::kInt64;
}

void validate_step(int64_t step) {
  TORCH_CHECK(step >= 0, "step must be non-negative");
}

void validate_probability(double probability) {
  TORCH_CHECK(std::isfinite(probability) && probability > 0.0 &&
                  probability <= 1.0,
              "mask probabilities must be finite and in (0,1]");
}

void validate_span_bounds(int64_t min_span, int64_t max_span) {
  TORCH_CHECK(min_span > 0 && max_span >= min_span,
              "span bounds must satisfy 0 < min_span <= max_span");
}

std::vector<int64_t> validate_feature_indices(
    const std::vector<int64_t>& indices,
    int64_t feature_dim,
    const std::string& name) {
  TORCH_CHECK(!indices.empty(), name, " must not be empty");
  std::unordered_set<int64_t> seen;
  seen.reserve(indices.size());
  for (const int64_t index : indices) {
    TORCH_CHECK(index >= 0 && index < feature_dim,
                name, " contains an out-of-range column");
    TORCH_CHECK(seen.insert(index).second,
                name, " contains a duplicate column");
  }
  return indices;
}

}  // namespace

double sample_mask_probability(
    const SpanMaskConfig& config,
    uint64_t seed,
    int64_t step) {
  validate_step(step);
  TORCH_CHECK(!config.probabilities.empty(),
              "the mask probability list must not be empty");
  for (const double probability : config.probabilities) {
    validate_probability(probability);
  }
  std::mt19937_64 random(
      splitmix64(seed ^ static_cast<uint64_t>(step)));
  std::uniform_int_distribution<std::size_t> distribution(
      0, config.probabilities.size() - 1);
  return config.probabilities[distribution(random)];
}

torch::Tensor make_span_mask(
    const torch::Tensor& valid,
    const torch::Tensor& sample_ids,
    double probability,
    int64_t min_span,
    int64_t max_span,
    uint64_t seed,
    int64_t step,
    uint64_t stream_key) {
  validate_probability(probability);
  validate_span_bounds(min_span, max_span);
  validate_step(step);
  TORCH_CHECK(valid.defined() && sample_ids.defined(),
              "valid and sample_ids must be defined");
  TORCH_CHECK(valid.device().is_cpu() && sample_ids.device().is_cpu(),
              "span masks must be generated on CPU");
  TORCH_CHECK(valid.dim() == 2 && valid.size(0) > 0 && valid.size(1) > 0,
              "valid must have a non-empty [B,L] shape");
  TORCH_CHECK(valid.scalar_type() == torch::kBool,
              "valid must be boolean");
  TORCH_CHECK(sample_ids.dim() == 1 && sample_ids.size(0) == valid.size(0),
              "sample_ids must have shape [B]");
  TORCH_CHECK(is_integer_type(sample_ids.scalar_type()),
              "sample_ids must be int32 or int64");

  auto contiguous_valid = valid.to(torch::kUInt8).contiguous();
  auto contiguous_ids = sample_ids.to(torch::kInt64).contiguous();
  auto mask = torch::zeros(
      valid.sizes(),
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
  auto valid_access = contiguous_valid.accessor<uint8_t, 2>();
  auto id_access = contiguous_ids.accessor<int64_t, 1>();
  auto mask_access = mask.accessor<bool, 2>();
  const int64_t batch_size = valid.size(0);
  const int64_t length = valid.size(1);

  for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    std::vector<int64_t> valid_positions;
    valid_positions.reserve(static_cast<std::size_t>(length));
    for (int64_t position = 0; position < length; ++position) {
      if (valid_access[batch_index][position]) {
        valid_positions.push_back(position);
      }
    }
    if (valid_positions.empty()) {
      continue;
    }

    const int64_t target_count = std::max<int64_t>(
        1,
        static_cast<int64_t>(
            std::llround(valid_positions.size() * probability)));
    const int64_t lower = valid_positions.front();
    const int64_t upper = valid_positions.back() + 1;
    uint64_t row_seed = seed;
    row_seed ^= splitmix64(static_cast<uint64_t>(step));
    row_seed ^= splitmix64(static_cast<uint64_t>(id_access[batch_index]));
    row_seed ^= splitmix64(static_cast<uint64_t>(batch_index));
    row_seed ^= stream_key;
    std::mt19937_64 random(splitmix64(row_seed));
    std::uniform_int_distribution<int64_t> span_distribution(
        min_span, max_span);
    int64_t covered = 0;
    const int64_t max_attempts = std::max<int64_t>(100, target_count * 20);
    for (int64_t attempt = 0;
         attempt < max_attempts && covered < target_count;
         ++attempt) {
      const int64_t span = std::min<int64_t>(
          span_distribution(random), upper - lower);
      std::uniform_int_distribution<int64_t> start_distribution(
          lower, upper - span);
      const int64_t start = start_distribution(random);
      for (int64_t position = start;
           position < start + span && covered < target_count;
           ++position) {
        if (valid_access[batch_index][position] &&
            !mask_access[batch_index][position]) {
          mask_access[batch_index][position] = true;
          ++covered;
        }
      }
    }
    if (covered < target_count) {
      std::shuffle(valid_positions.begin(), valid_positions.end(), random);
      for (const int64_t position : valid_positions) {
        if (!mask_access[batch_index][position]) {
          mask_access[batch_index][position] = true;
          if (++covered >= target_count) {
            break;
          }
        }
      }
    }
  }
  return mask;
}

StreamBatch mask_span_features(
    const StreamBatch& stream,
    const torch::Tensor& mask,
    const SpanMaskConfig& config) {
  TORCH_CHECK(stream.features.defined() && stream.valid.defined(),
              "stream features and validity must be defined");
  TORCH_CHECK(stream.features.dim() == 3 && stream.features.size(0) > 0 &&
                  stream.features.size(1) > 0 &&
                  stream.features.size(2) > 0,
              "stream features must have a non-empty [B,L,F] shape");
  TORCH_CHECK(stream.features.is_floating_point(),
              "stream features must be floating point");
  TORCH_CHECK(stream.valid.dim() == 2 &&
                  stream.valid.sizes() == stream.features.sizes().slice(0, 2) &&
                  stream.valid.scalar_type() == torch::kBool,
              "stream validity must be boolean with shape [B,L]");
  TORCH_CHECK(mask.defined() && mask.dim() == 2 &&
                  mask.sizes() == stream.valid.sizes() &&
                  mask.scalar_type() == torch::kBool,
              "mask must be boolean with shape [B,L]");
  TORCH_CHECK(stream.features.device() == stream.valid.device() &&
                  stream.features.device() == mask.device(),
              "features, validity, and mask must be on one device");
  TORCH_CHECK(!mask.logical_and(stream.valid.logical_not()).any().item<bool>(),
              "the span mask may select only valid tokens");
  TORCH_CHECK(config.indicator_feature_index >= 0 &&
                  config.indicator_feature_index < stream.features.size(2),
              "indicator_feature_index is out of range");
  const auto masked_indices = validate_feature_indices(
      config.masked_feature_indices,
      stream.features.size(2),
      "masked_feature_indices");
  TORCH_CHECK(
      std::find(
          masked_indices.begin(),
          masked_indices.end(),
          config.indicator_feature_index) == masked_indices.end(),
      "indicator_feature_index must not be a masked feature column");

  StreamBatch result = stream;
  result.features = stream.features.clone();
  for (const int64_t feature_index : masked_indices) {
    result.features.select(-1, feature_index).masked_fill_(mask, 0.0);
  }
  result.features.select(-1, config.indicator_feature_index)
      .copy_(mask.to(result.features.scalar_type()));
  return result;
}

ReconstructionHeadImpl::ReconstructionHeadImpl(
    int64_t model_dim,
    int64_t feature_dim,
    double dropout) {
  TORCH_CHECK(model_dim > 0, "model_dim must be positive");
  TORCH_CHECK(feature_dim > 0, "feature_dim must be positive");
  TORCH_CHECK(std::isfinite(dropout) && dropout >= 0.0 && dropout < 1.0,
              "dropout must be finite and in [0,1)");
  shared = register_module(
      "shared",
      torch::nn::Sequential(
          torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({model_dim})),
          torch::nn::Linear(model_dim, model_dim),
          torch::nn::GELU(),
          torch::nn::Dropout(dropout)));
  reconstruction = register_module(
      "reconstruction", torch::nn::Linear(model_dim, feature_dim));
}

torch::Tensor ReconstructionHeadImpl::forward(
    const torch::Tensor& memory) {
  TORCH_CHECK(memory.defined() && memory.dim() == 3 &&
                  memory.size(0) > 0 && memory.size(1) > 0,
              "memory must have a non-empty [B,L,D] shape");
  TORCH_CHECK(memory.is_floating_point(),
              "memory must be floating point");
  TORCH_CHECK(memory.size(2) == shared->ptr<torch::nn::LayerNormImpl>(0)
                                     ->options.normalized_shape()[0],
              "memory has the wrong model dimension");
  TORCH_CHECK(memory.device() == reconstruction->weight.device() &&
                  memory.scalar_type() == reconstruction->weight.scalar_type(),
              "memory device and dtype must match head parameters");
  return reconstruction->forward(shared->forward(memory));
}

torch::Tensor masked_reconstruction_loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& mask,
    const std::vector<int64_t>& feature_indices) {
  TORCH_CHECK(prediction.defined() && target.defined() && mask.defined(),
              "prediction, target, and mask must be defined");
  TORCH_CHECK(prediction.dim() == 3 && target.dim() == 3,
              "prediction and target must have shape [B,L,F]");
  TORCH_CHECK(prediction.size(0) == target.size(0) &&
                  prediction.size(1) == target.size(1),
              "prediction and target batch/length dimensions must match");
  TORCH_CHECK(prediction.is_floating_point() && target.is_floating_point(),
              "prediction and target must be floating point");
  TORCH_CHECK(prediction.device() == target.device(),
              "prediction and target must be on one device");
  TORCH_CHECK(mask.dim() == 2 && mask.size(0) == prediction.size(0) &&
                  mask.size(1) == prediction.size(1) &&
                  mask.scalar_type() == torch::kBool &&
                  mask.device() == prediction.device(),
              "mask must be boolean with shape [B,L] on the data device");
  validate_feature_indices(
      feature_indices,
      prediction.size(2),
      "feature_indices for prediction");
  validate_feature_indices(
      feature_indices,
      target.size(2),
      "feature_indices for target");

  auto index = torch::tensor(
      feature_indices,
      torch::TensorOptions()
          .device(prediction.device())
          .dtype(torch::kLong));
  auto prediction_selected = prediction.index_select(-1, index).to(torch::kFloat32);
  auto target_selected = target.index_select(-1, index).to(torch::kFloat32);
  auto expanded_mask = mask.unsqueeze(-1).expand_as(prediction_selected);
  // Clear excluded positions before arithmetic so NaN padding outside the
  // included elements cannot contaminate an otherwise finite objective.
  prediction_selected = torch::where(
      expanded_mask,
      prediction_selected,
      torch::zeros_like(prediction_selected));
  target_selected = torch::where(
      expanded_mask,
      target_selected,
      torch::zeros_like(target_selected));
  TORCH_CHECK(torch::isfinite(prediction_selected).all().item<bool>(),
              "prediction is non-finite at a selected masked element");
  TORCH_CHECK(torch::isfinite(target_selected).all().item<bool>(),
              "target is non-finite at a selected masked element");
  auto difference = prediction_selected - target_selected;
  auto denominator = expanded_mask.sum().to(torch::kFloat32).clamp_min(1.0);
  return difference.square().sum() / denominator;
}

torch::Tensor mean_stream_loss(
    const std::vector<torch::Tensor>& losses) {
  TORCH_CHECK(!losses.empty(), "losses must not be empty");
  TORCH_CHECK(losses.front().defined() && losses.front().numel() == 1 &&
                  losses.front().is_floating_point(),
              "each stream loss must be a floating scalar");
  auto total = losses.front();
  TORCH_CHECK(torch::isfinite(total).item<bool>(),
              "stream losses must be finite");
  for (std::size_t index = 1; index < losses.size(); ++index) {
    const auto& loss = losses[index];
    TORCH_CHECK(loss.defined() && loss.numel() == 1 &&
                    loss.is_floating_point(),
                "each stream loss must be a floating scalar");
    TORCH_CHECK(loss.device() == total.device() &&
                    loss.scalar_type() == total.scalar_type(),
                "stream losses must share a device and dtype");
    TORCH_CHECK(torch::isfinite(loss).item<bool>(),
                "stream losses must be finite");
    total = total + loss;
  }
  return total / static_cast<double>(losses.size());
}

}  // namespace cmt
