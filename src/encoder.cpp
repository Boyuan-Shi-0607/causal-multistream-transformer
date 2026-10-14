#include "cmt/encoder.hpp"

#include <ATen/ops/scaled_dot_product_attention.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cmt {
namespace {

constexpr double kMaximumAbsolutePosition = 1'000'000.0;

bool is_supported_position_type(torch::ScalarType type) {
  return type == torch::kInt32 || type == torch::kInt64 ||
         type == torch::kFloat32 || type == torch::kFloat64;
}

bool is_integer_metadata_type(torch::ScalarType type) {
  return type == torch::kInt32 || type == torch::kInt64;
}

void validate_config(const EncoderConfig& config) {
  TORCH_CHECK(config.input_dim > 0, "input_dim must be positive");
  TORCH_CHECK(config.model_dim > 0, "model_dim must be positive");
  TORCH_CHECK(config.heads > 0, "heads must be positive");
  TORCH_CHECK(config.model_dim % config.heads == 0,
              "model_dim must be divisible by heads");
  TORCH_CHECK((config.model_dim / config.heads) % 2 == 0,
              "each attention head must have an even dimension for RoPE");
  TORCH_CHECK(config.context_layers > 0, "context_layers must be positive");
  TORCH_CHECK(config.target_layers > 0, "target_layers must be positive");
  TORCH_CHECK(config.ffn_multiplier > 0, "ffn_multiplier must be positive");
  TORCH_CHECK(config.stream_count > 0, "stream_count must be positive");
  TORCH_CHECK(config.scale_count > 0, "scale_count must be positive");
  TORCH_CHECK(config.segment_count > 0, "segment_count must be positive");
  TORCH_CHECK(std::isfinite(config.dropout) && config.dropout >= 0.0 &&
                  config.dropout < 1.0,
              "dropout must be finite and in [0,1)");
  TORCH_CHECK(std::isfinite(config.rope_base) && config.rope_base > 1.0,
              "rope_base must be finite and greater than one");
}

void validate_rank_two(
    const torch::Tensor& tensor,
    int64_t batch,
    int64_t length,
    const std::string& name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == batch &&
                  tensor.size(1) == length,
              name, " must have shape [B,L]");
}

void validate_position_values(
    const torch::Tensor& positions,
    const std::string& name) {
  TORCH_CHECK(is_supported_position_type(positions.scalar_type()),
              name,
              " must be int32, int64, float32, or float64; half-precision "
              "positions are not accepted");
  const auto as_double = positions.to(torch::kFloat64);
  TORCH_CHECK(torch::isfinite(as_double).all().item<bool>(),
              name, " contains non-finite values");
  TORCH_CHECK(as_double.abs().max().item<double>() <= kMaximumAbsolutePosition,
              name,
              " exceeds the supported relative-position range; subtract a "
              "common origin before calling the encoder");
}

torch::Tensor rotate_half(const torch::Tensor& value) {
  const int64_t dimension = value.size(-1);
  auto even = value.slice(-1, 0, dimension, 2);
  auto odd = value.slice(-1, 1, dimension, 2);
  return torch::stack({-odd, even}, -1).flatten(-2);
}

std::pair<torch::Tensor, torch::Tensor> rotary_factors(
    const torch::Tensor& positions,
    int64_t head_dim,
    double base,
    torch::ScalarType output_type) {
  validate_position_values(positions, "positions");
  // Integer positions are converted directly to float64. This avoids first
  // rounding large relative offsets to float32 and losing adjacent positions.
  const auto position64 = positions.to(torch::kFloat64);
  const auto options = torch::TensorOptions()
                           .device(positions.device())
                           .dtype(torch::kFloat64);
  auto index = torch::arange(0, head_dim, 2, options);
  auto inverse_frequency = torch::pow(
      torch::full_like(index, base),
      -index / static_cast<double>(head_dim));
  auto phase = position64.unsqueeze(-1) *
               inverse_frequency.view({1, 1, -1});
  auto repeated = torch::repeat_interleave(phase, 2, -1);
  return {
      repeated.cos().to(output_type).unsqueeze(1),
      repeated.sin().to(output_type).unsqueeze(1)};
}

torch::Tensor apply_rotary(
    const torch::Tensor& value,
    const torch::Tensor& cosine,
    const torch::Tensor& sine) {
  return value * cosine + rotate_half(value) * sine;
}

torch::Tensor zero_invalid(
    const torch::Tensor& value,
    const torch::Tensor& valid) {
  return torch::where(
      valid.unsqueeze(-1), value, torch::zeros_like(value));
}

torch::nn::Sequential make_feed_forward(const EncoderConfig& config) {
  return torch::nn::Sequential(
      torch::nn::Linear(
          config.model_dim,
          config.model_dim * config.ffn_multiplier),
      torch::nn::GELU(),
      torch::nn::Dropout(config.dropout),
      torch::nn::Linear(
          config.model_dim * config.ffn_multiplier,
          config.model_dim),
      torch::nn::Dropout(config.dropout));
}

struct ContextBlockImpl : torch::nn::Module {
  explicit ContextBlockImpl(const EncoderConfig& config) {
    attention = register_module(
        "attention",
        RotaryAttention(
            config.model_dim,
            config.heads,
            config.dropout,
            config.rope_base));
    attention_norm = register_module(
        "attention_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.model_dim})));
    feed_forward_norm = register_module(
        "feed_forward_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.model_dim})));
    feed_forward = register_module(
        "feed_forward", make_feed_forward(config));
  }

  torch::Tensor forward(
      const torch::Tensor& hidden,
      const torch::Tensor& positions,
      const torch::Tensor& allow) {
    auto normalized = attention_norm->forward(hidden);
    auto result = hidden + attention->forward(
                               normalized,
                               normalized,
                               positions,
                               positions,
                               allow);
    return result + feed_forward->forward(feed_forward_norm->forward(result));
  }

  RotaryAttention attention{nullptr};
  torch::nn::LayerNorm attention_norm{nullptr};
  torch::nn::LayerNorm feed_forward_norm{nullptr};
  torch::nn::Sequential feed_forward{nullptr};
};
TORCH_MODULE(ContextBlock);

struct TargetBlockImpl : torch::nn::Module {
  explicit TargetBlockImpl(const EncoderConfig& config) {
    self_attention = register_module(
        "self_attention",
        RotaryAttention(
            config.model_dim,
            config.heads,
            config.dropout,
            config.rope_base));
    cross_attention = register_module(
        "cross_attention",
        RotaryAttention(
            config.model_dim,
            config.heads,
            config.dropout,
            config.rope_base));
    self_norm = register_module(
        "self_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.model_dim})));
    cross_query_norm = register_module(
        "cross_query_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.model_dim})));
    cross_key_value_norm = register_module(
        "cross_key_value_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.model_dim})));
    feed_forward_norm = register_module(
        "feed_forward_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.model_dim})));
    feed_forward = register_module(
        "feed_forward", make_feed_forward(config));
  }

  torch::Tensor forward(
      const torch::Tensor& target,
      const torch::Tensor& context,
      const torch::Tensor& target_positions,
      const torch::Tensor& context_positions,
      const torch::Tensor& self_allow,
      const torch::Tensor& cross_allow) {
    auto self_normalized = self_norm->forward(target);
    auto result = target + self_attention->forward(
                               self_normalized,
                               self_normalized,
                               target_positions,
                               target_positions,
                               self_allow);
    result = result + cross_attention->forward(
                          cross_query_norm->forward(result),
                          cross_key_value_norm->forward(context),
                          target_positions,
                          context_positions,
                          cross_allow);
    return result + feed_forward->forward(feed_forward_norm->forward(result));
  }

  RotaryAttention self_attention{nullptr};
  RotaryAttention cross_attention{nullptr};
  torch::nn::LayerNorm self_norm{nullptr};
  torch::nn::LayerNorm cross_query_norm{nullptr};
  torch::nn::LayerNorm cross_key_value_norm{nullptr};
  torch::nn::LayerNorm feed_forward_norm{nullptr};
  torch::nn::Sequential feed_forward{nullptr};
};
TORCH_MODULE(TargetBlock);

struct PreparedStream {
  torch::Tensor features;
  torch::Tensor positions;
  torch::Tensor available_at;
  torch::Tensor valid;
  torch::Tensor segment_ids;
  int64_t stream_id = 0;
  int64_t scale_id = 0;
};

PreparedStream prepare_stream(
    const StreamBatch& stream,
    const EncoderConfig& config,
    const std::string& name,
    c10::optional<int64_t> expected_batch,
    c10::optional<torch::Device> expected_device,
    c10::optional<torch::ScalarType> expected_feature_type,
    c10::optional<torch::ScalarType> expected_position_type,
    c10::optional<torch::ScalarType> expected_time_type) {
  TORCH_CHECK(stream.features.defined(), name, ".features must be defined");
  TORCH_CHECK(stream.features.dim() == 3,
              name, ".features must have shape [B,L,F]");
  const int64_t batch = stream.features.size(0);
  const int64_t length = stream.features.size(1);
  TORCH_CHECK(batch > 0 && length > 0,
              name, " must have positive batch and sequence dimensions");
  TORCH_CHECK(stream.features.size(2) == config.input_dim,
              name, ".features has the wrong feature dimension");
  TORCH_CHECK(stream.features.is_floating_point(),
              name, ".features must be floating point");
  TORCH_CHECK(!expected_batch.has_value() || batch == *expected_batch,
              name, " has a different batch size");
  TORCH_CHECK(!expected_device.has_value() ||
                  stream.features.device() == *expected_device,
              name, " is on a different device");
  TORCH_CHECK(!expected_feature_type.has_value() ||
                  stream.features.scalar_type() == *expected_feature_type,
              name, " has a different feature dtype");

  validate_rank_two(stream.positions, batch, length, name + ".positions");
  validate_rank_two(
      stream.available_at, batch, length, name + ".available_at");
  validate_rank_two(stream.valid, batch, length, name + ".valid");
  TORCH_CHECK(stream.positions.device() == stream.features.device() &&
                  stream.available_at.device() == stream.features.device() &&
                  stream.valid.device() == stream.features.device(),
              name, " tensors must be on one device");
  TORCH_CHECK(is_supported_position_type(stream.positions.scalar_type()),
              name,
              ".positions must be int32, int64, float32, or float64");
  TORCH_CHECK(!expected_position_type.has_value() ||
                  stream.positions.scalar_type() == *expected_position_type,
              name, " has a different position dtype");
  TORCH_CHECK(is_integer_metadata_type(stream.available_at.scalar_type()),
              name, ".available_at must be int32 or int64");
  TORCH_CHECK(!expected_time_type.has_value() ||
                  stream.available_at.scalar_type() == *expected_time_type,
              name, " has a different available_at dtype");
  TORCH_CHECK(stream.valid.scalar_type() == torch::kBool,
              name, ".valid must be boolean");
  TORCH_CHECK(stream.stream_id >= 0 &&
                  stream.stream_id < config.stream_count,
              name, ".stream_id is out of range");
  TORCH_CHECK(stream.scale_id >= 0 &&
                  stream.scale_id < config.scale_count,
              name, ".scale_id is out of range");

  torch::Tensor segments;
  if (stream.segment_ids.defined()) {
    validate_rank_two(
        stream.segment_ids, batch, length, name + ".segment_ids");
    TORCH_CHECK(stream.segment_ids.device() == stream.features.device(),
                name, ".segment_ids is on a different device");
    TORCH_CHECK(is_integer_metadata_type(stream.segment_ids.scalar_type()),
                name, ".segment_ids must be int32 or int64");
    segments = torch::where(
                   stream.valid,
                   stream.segment_ids,
                   torch::zeros_like(stream.segment_ids))
                   .to(torch::kLong);
  } else {
    segments = torch::zeros(
        {batch, length},
        torch::TensorOptions()
            .device(stream.features.device())
            .dtype(torch::kLong));
  }
  TORCH_CHECK(segments.min().item<int64_t>() >= 0 &&
                  segments.max().item<int64_t>() < config.segment_count,
              name, ".segment_ids is out of range on a valid token");

  auto finite_or_invalid = torch::isfinite(stream.features).logical_or(
      stream.valid.logical_not().unsqueeze(-1));
  TORCH_CHECK(finite_or_invalid.all().item<bool>(),
              name, ".features is non-finite on a valid token");

  auto features = torch::where(
      stream.valid.unsqueeze(-1),
      stream.features,
      torch::zeros_like(stream.features));
  auto positions = torch::where(
      stream.valid, stream.positions, torch::zeros_like(stream.positions));
  auto available_at = torch::where(
      stream.valid,
      stream.available_at,
      torch::zeros_like(stream.available_at));
  validate_position_values(positions, name + ".positions");

  return {
      features,
      positions,
      available_at,
      stream.valid,
      segments,
      stream.stream_id,
      stream.scale_id};
}

}  // namespace

torch::Tensor make_causal_mask(
    const torch::Tensor& query_available_at,
    const torch::Tensor& key_available_at,
    const torch::Tensor& query_valid,
    const torch::Tensor& key_valid) {
  TORCH_CHECK(query_available_at.defined() && key_available_at.defined() &&
                  query_valid.defined() && key_valid.defined(),
              "causal-mask inputs must be defined");
  TORCH_CHECK(query_available_at.dim() == 2 && key_available_at.dim() == 2 &&
                  query_valid.dim() == 2 && key_valid.dim() == 2,
              "causal-mask inputs must have shape [B,L]");
  TORCH_CHECK(query_available_at.sizes() == query_valid.sizes() &&
                  key_available_at.sizes() == key_valid.sizes(),
              "causal-mask timestamp and validity shapes differ");
  TORCH_CHECK(query_available_at.size(0) == key_available_at.size(0),
              "causal-mask batch sizes differ");
  TORCH_CHECK(query_available_at.device() == key_available_at.device() &&
                  query_available_at.device() == query_valid.device() &&
                  query_available_at.device() == key_valid.device(),
              "causal-mask inputs must be on one device");
  TORCH_CHECK(is_integer_metadata_type(query_available_at.scalar_type()) &&
                  query_available_at.scalar_type() ==
                      key_available_at.scalar_type(),
              "available_at tensors must share an int32 or int64 dtype");
  TORCH_CHECK(query_valid.scalar_type() == torch::kBool &&
                  key_valid.scalar_type() == torch::kBool,
              "causal-mask validity tensors must be boolean");

  return key_available_at.unsqueeze(1)
      .le(query_available_at.unsqueeze(2))
      .logical_and(query_valid.unsqueeze(2))
      .logical_and(key_valid.unsqueeze(1));
}

RotaryAttentionImpl::RotaryAttentionImpl(
    int64_t model_dim,
    int64_t heads_,
    double dropout_,
    double rope_base_)
    : heads(heads_),
      head_dim(heads_ > 0 ? model_dim / heads_ : 0),
      dropout(dropout_),
      rope_base(rope_base_) {
  TORCH_CHECK(model_dim > 0 && heads > 0 && model_dim % heads == 0,
              "model_dim must be positive and divisible by heads");
  TORCH_CHECK(head_dim % 2 == 0,
              "each attention head must have an even dimension for RoPE");
  TORCH_CHECK(std::isfinite(dropout) && dropout >= 0.0 && dropout < 1.0,
              "dropout must be finite and in [0,1)");
  TORCH_CHECK(std::isfinite(rope_base) && rope_base > 1.0,
              "rope_base must be finite and greater than one");
  query_projection = register_module(
      "query_projection", torch::nn::Linear(model_dim, model_dim));
  key_projection = register_module(
      "key_projection", torch::nn::Linear(model_dim, model_dim));
  value_projection = register_module(
      "value_projection", torch::nn::Linear(model_dim, model_dim));
  output_projection = register_module(
      "output_projection", torch::nn::Linear(model_dim, model_dim));
}

torch::Tensor RotaryAttentionImpl::forward(
    const torch::Tensor& query,
    const torch::Tensor& key_value,
    const torch::Tensor& query_positions,
    const torch::Tensor& key_positions,
    const torch::Tensor& allow) {
  TORCH_CHECK(query.defined() && key_value.defined(),
              "query and key_value must be defined");
  TORCH_CHECK(query.dim() == 3 && key_value.dim() == 3,
              "query and key_value must have shape [B,L,D]");
  TORCH_CHECK(query.size(0) == key_value.size(0) &&
                  query.size(2) == heads * head_dim &&
                  key_value.size(2) == heads * head_dim,
              "attention tensor shapes are incompatible");
  TORCH_CHECK(query.is_floating_point() && key_value.is_floating_point() &&
                  query.scalar_type() == key_value.scalar_type(),
              "query and key_value must share a floating dtype");
  TORCH_CHECK(query.device() == key_value.device(),
              "query and key_value must be on one device");
  const int64_t batch = query.size(0);
  const int64_t query_length = query.size(1);
  const int64_t key_length = key_value.size(1);
  validate_rank_two(
      query_positions,
      batch,
      query_length,
      "query_positions");
  validate_rank_two(
      key_positions,
      batch,
      key_length,
      "key_positions");
  TORCH_CHECK(query_positions.device() == query.device() &&
                  key_positions.device() == query.device(),
              "attention positions must be on the query device");
  validate_position_values(query_positions, "query_positions");
  validate_position_values(key_positions, "key_positions");
  TORCH_CHECK(allow.defined() && allow.scalar_type() == torch::kBool &&
                  allow.dim() == 3 && allow.size(0) == batch &&
                  allow.size(1) == query_length &&
                  allow.size(2) == key_length,
              "allow must be boolean with shape [B,Q,K]");
  TORCH_CHECK(allow.device() == query.device(),
              "allow must be on the query device");
  TORCH_CHECK(query.scalar_type() == query_projection->weight.scalar_type(),
              "attention input dtype must match module parameters");

  auto q = query_projection->forward(query)
               .view({batch, query_length, heads, head_dim})
               .transpose(1, 2);
  auto k = key_projection->forward(key_value)
               .view({batch, key_length, heads, head_dim})
               .transpose(1, 2);
  auto v = value_projection->forward(key_value)
               .view({batch, key_length, heads, head_dim})
               .transpose(1, 2);
  const auto query_factors = rotary_factors(
      query_positions, head_dim, rope_base, q.scalar_type());
  const auto key_factors = rotary_factors(
      key_positions, head_dim, rope_base, k.scalar_type());
  q = apply_rotary(q, query_factors.first, query_factors.second);
  k = apply_rotary(k, key_factors.first, key_factors.second);

  auto expanded_allow = allow.unsqueeze(1);
  auto additive_mask = torch::zeros(
      {batch, 1, query_length, key_length}, q.options());
  additive_mask.masked_fill_(
      expanded_allow.logical_not(),
      -std::numeric_limits<double>::infinity());
  auto row_has_key = expanded_allow.any(-1, true);
  // Some scaled-dot-product-attention backends return NaN for an all-masked
  // row. Temporarily make only those internal rows finite, then discard their
  // values and projection bias exactly below.
  additive_mask = torch::where(
      row_has_key, additive_mask, torch::zeros_like(additive_mask));
  q = torch::where(row_has_key, q, torch::zeros_like(q));
  c10::optional<at::Tensor> mask = additive_mask;
  const double dropout_probability =
      is_training() ? dropout : 0.0;
  auto attended = at::scaled_dot_product_attention(
      q, k, v, mask, dropout_probability, false);
  attended = torch::where(
      row_has_key, attended, torch::zeros_like(attended));
  attended = attended.transpose(1, 2)
                 .contiguous()
                 .view({batch, query_length, heads * head_dim});
  auto output = output_projection->forward(attended);
  return torch::where(
      row_has_key.any(1), output, torch::zeros_like(output));
}

CausalMultiStreamEncoderImpl::CausalMultiStreamEncoderImpl(
    const EncoderConfig& config_)
    : config(config_) {
  validate_config(config);
  input_projection = register_module(
      "input_projection",
      torch::nn::Linear(config.input_dim, config.model_dim));
  stream_embedding = register_module(
      "stream_embedding",
      torch::nn::Embedding(config.stream_count, config.model_dim));
  scale_embedding = register_module(
      "scale_embedding",
      torch::nn::Embedding(config.scale_count, config.model_dim));
  segment_embedding = register_module(
      "segment_embedding",
      torch::nn::Embedding(config.segment_count, config.model_dim));
  context_blocks = register_module(
      "context_blocks", torch::nn::ModuleList());
  target_blocks = register_module(
      "target_blocks", torch::nn::ModuleList());
  for (int64_t layer = 0; layer < config.context_layers; ++layer) {
    context_blocks->push_back(ContextBlock(config));
  }
  for (int64_t layer = 0; layer < config.target_layers; ++layer) {
    target_blocks->push_back(TargetBlock(config));
  }
  context_norm = register_module(
      "context_norm",
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({config.model_dim})));
  target_norm = register_module(
      "target_norm",
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({config.model_dim})));
}

EncoderOutput CausalMultiStreamEncoderImpl::forward(
    const StreamBatch& target,
    const std::vector<StreamBatch>& contexts) {
  TORCH_CHECK(!contexts.empty(), "at least one context stream is required");
  auto prepared_target = prepare_stream(
      target,
      config,
      "target",
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt);
  TORCH_CHECK(prepared_target.features.scalar_type() ==
                  input_projection->weight.scalar_type(),
              "feature dtype must match encoder parameters");

  auto embed = [this](const PreparedStream& stream) {
    const int64_t batch = stream.features.size(0);
    const int64_t length = stream.features.size(1);
    auto id_options = torch::TensorOptions()
                          .device(stream.features.device())
                          .dtype(torch::kLong);
    auto stream_ids = torch::full(
        {batch, length}, stream.stream_id, id_options);
    auto scale_ids = torch::full(
        {batch, length}, stream.scale_id, id_options);
    auto hidden = input_projection->forward(stream.features) +
                  stream_embedding->forward(stream_ids) +
                  scale_embedding->forward(scale_ids) +
                  segment_embedding->forward(stream.segment_ids);
    return zero_invalid(hidden, stream.valid);
  };

  std::vector<torch::Tensor> context_memories;
  context_memories.reserve(contexts.size());
  std::vector<torch::Tensor> context_positions;
  std::vector<torch::Tensor> context_available_at;
  std::vector<torch::Tensor> context_valid;
  for (std::size_t index = 0; index < contexts.size(); ++index) {
    auto prepared = prepare_stream(
        contexts[index],
        config,
        "contexts[" + std::to_string(index) + "]",
        prepared_target.features.size(0),
        prepared_target.features.device(),
        prepared_target.features.scalar_type(),
        prepared_target.positions.scalar_type(),
        prepared_target.available_at.scalar_type());
    auto hidden = embed(prepared);
    auto allow = make_causal_mask(
        prepared.available_at,
        prepared.available_at,
        prepared.valid,
        prepared.valid);
    for (auto& module : *context_blocks) {
      auto block = std::dynamic_pointer_cast<ContextBlockImpl>(module);
      TORCH_CHECK(block != nullptr, "invalid context block module");
      hidden = block->forward(hidden, prepared.positions, allow);
      hidden = zero_invalid(hidden, prepared.valid);
    }
    hidden = zero_invalid(context_norm->forward(hidden), prepared.valid);
    context_memories.push_back(hidden);
    context_positions.push_back(prepared.positions);
    context_available_at.push_back(prepared.available_at);
    context_valid.push_back(prepared.valid);
  }

  auto joined_context = torch::cat(context_memories, 1);
  auto joined_positions = torch::cat(context_positions, 1);
  auto joined_available_at = torch::cat(context_available_at, 1);
  auto joined_valid = torch::cat(context_valid, 1);

  auto target_hidden = embed(prepared_target);
  auto target_allow = make_causal_mask(
      prepared_target.available_at,
      prepared_target.available_at,
      prepared_target.valid,
      prepared_target.valid);
  auto cross_allow = make_causal_mask(
      prepared_target.available_at,
      joined_available_at,
      prepared_target.valid,
      joined_valid);
  for (auto& module : *target_blocks) {
    auto block = std::dynamic_pointer_cast<TargetBlockImpl>(module);
    TORCH_CHECK(block != nullptr, "invalid target block module");
    target_hidden = block->forward(
        target_hidden,
        joined_context,
        prepared_target.positions,
        joined_positions,
        target_allow,
        cross_allow);
    target_hidden = zero_invalid(target_hidden, prepared_target.valid);
  }
  target_hidden = zero_invalid(
      target_norm->forward(target_hidden), prepared_target.valid);

  return {
      target_hidden,
      joined_context,
      std::move(context_memories)};
}

}  // namespace cmt
