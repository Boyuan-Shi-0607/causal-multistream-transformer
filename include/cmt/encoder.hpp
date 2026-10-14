#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace cmt {

struct EncoderConfig {
  int64_t input_dim = 8;
  int64_t model_dim = 64;
  int64_t heads = 4;
  int64_t context_layers = 2;
  int64_t target_layers = 2;
  int64_t ffn_multiplier = 4;
  int64_t stream_count = 3;
  int64_t scale_count = 3;
  int64_t segment_count = 1;
  double dropout = 0.0;
  double rope_base = 10000.0;
};

struct StreamBatch {
  torch::Tensor features;
  torch::Tensor positions;
  torch::Tensor available_at;
  torch::Tensor valid;
  torch::Tensor segment_ids;
  int64_t stream_id = 0;
  int64_t scale_id = 0;
};

struct EncoderOutput {
  torch::Tensor target_memory;
  torch::Tensor context_memory;
  std::vector<torch::Tensor> context_memories;
};

// Returns a boolean [B,Q,K] mask. A key is visible only when both tokens are
// valid and the key was available no later than the query.
torch::Tensor make_causal_mask(
    const torch::Tensor& query_available_at,
    const torch::Tensor& key_available_at,
    const torch::Tensor& query_valid,
    const torch::Tensor& key_valid);

struct RotaryAttentionImpl : torch::nn::Module {
  RotaryAttentionImpl(
      int64_t model_dim,
      int64_t heads,
      double dropout = 0.0,
      double rope_base = 10000.0);

  torch::Tensor forward(
      const torch::Tensor& query,
      const torch::Tensor& key_value,
      const torch::Tensor& query_positions,
      const torch::Tensor& key_positions,
      const torch::Tensor& allow);

  int64_t heads = 0;
  int64_t head_dim = 0;
  double dropout = 0.0;
  double rope_base = 10000.0;
  torch::nn::Linear query_projection{nullptr};
  torch::nn::Linear key_projection{nullptr};
  torch::nn::Linear value_projection{nullptr};
  torch::nn::Linear output_projection{nullptr};
};
TORCH_MODULE(RotaryAttention);

struct CausalMultiStreamEncoderImpl : torch::nn::Module {
  explicit CausalMultiStreamEncoderImpl(const EncoderConfig& config);

  EncoderOutput forward(
      const StreamBatch& target,
      const std::vector<StreamBatch>& contexts);

  EncoderConfig config;
  torch::nn::Linear input_projection{nullptr};
  torch::nn::Embedding stream_embedding{nullptr};
  torch::nn::Embedding scale_embedding{nullptr};
  torch::nn::Embedding segment_embedding{nullptr};
  torch::nn::ModuleList context_blocks{nullptr};
  torch::nn::ModuleList target_blocks{nullptr};
  torch::nn::LayerNorm context_norm{nullptr};
  torch::nn::LayerNorm target_norm{nullptr};
};
TORCH_MODULE(CausalMultiStreamEncoder);

}  // namespace cmt
