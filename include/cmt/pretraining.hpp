#pragma once

#include "cmt/encoder.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace cmt {

struct SpanMaskConfig {
  std::vector<double> probabilities;
  int64_t min_span = 1;
  int64_t max_span = 1;
  std::vector<int64_t> masked_feature_indices;
  int64_t indicator_feature_index = -1;
};

double sample_mask_probability(
    const SpanMaskConfig& config,
    uint64_t seed,
    int64_t step);

torch::Tensor make_span_mask(
    const torch::Tensor& valid,
    const torch::Tensor& sample_ids,
    double probability,
    int64_t min_span,
    int64_t max_span,
    uint64_t seed,
    int64_t step,
    uint64_t stream_key);

StreamBatch mask_span_features(
    const StreamBatch& stream,
    const torch::Tensor& mask,
    const SpanMaskConfig& config);

struct ReconstructionHeadImpl : torch::nn::Module {
  ReconstructionHeadImpl(
      int64_t model_dim,
      int64_t feature_dim,
      double dropout = 0.0);

  torch::Tensor forward(const torch::Tensor& memory);

  torch::nn::Sequential shared{nullptr};
  torch::nn::Linear reconstruction{nullptr};
};
TORCH_MODULE(ReconstructionHead);

torch::Tensor masked_reconstruction_loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& mask,
    const std::vector<int64_t>& feature_indices);

torch::Tensor mean_stream_loss(
    const std::vector<torch::Tensor>& losses);

}  // namespace cmt
