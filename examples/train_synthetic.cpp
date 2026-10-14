#include <cmt/pretraining.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TrainingModelImpl : torch::nn::Module {
  explicit TrainingModelImpl(const cmt::EncoderConfig& config) {
    encoder = register_module("encoder", cmt::CausalMultiStreamEncoder(config));
    reconstruction = register_module(
        "reconstruction", cmt::ReconstructionHead(
                              config.model_dim, config.input_dim, config.dropout));
  }
  cmt::CausalMultiStreamEncoder encoder{nullptr};
  cmt::ReconstructionHead reconstruction{nullptr};
};
TORCH_MODULE(TrainingModel);

struct SyntheticBatch {
  cmt::StreamBatch target;
  std::vector<cmt::StreamBatch> contexts;
  torch::Tensor sample_ids;
};

cmt::StreamBatch signal_stream(const torch::Tensor& ids, int64_t length,
                               int64_t stride, int64_t first, int64_t id) {
  const auto positions = (torch::arange(length, torch::kInt64) * stride + first)
                             .unsqueeze(0).repeat({ids.size(0), 1});
  const auto phase = positions.to(torch::kFloat32) * 0.08 +
                     ids.to(torch::kFloat32).unsqueeze(1) * 0.37;
  const auto a = phase.sin();
  const auto b = phase.cos();
  const auto first_signal = 0.8 * a + 0.2 * b;
  const auto second_signal = 0.4 * a - 0.7 * b;
  // Redundant generated channels make this a small, learnable integration
  // example. The last channel is reserved for the masking indicator.
  cmt::StreamBatch stream;
  stream.features = torch::stack(
      {first_signal, second_signal, 0.5 * a * b,
       first_signal - second_signal, a, b, (phase * 0.5).sin(),
       torch::ones_like(a), torch::zeros_like(a)}, -1);
  stream.positions = positions;
  stream.available_at = positions.clone();
  stream.valid = torch::ones(positions.sizes(), torch::kBool);
  stream.stream_id = id;
  stream.scale_id = id;
  return stream;
}

SyntheticBatch signals(int64_t first_id) {
  SyntheticBatch batch;
  batch.sample_ids = torch::arange(first_id, first_id + 8, torch::kInt64);
  batch.target = signal_stream(batch.sample_ids, 24, 1, 0, 0);
  batch.contexts = {
      signal_stream(batch.sample_ids, 6, 9, -18, 2),
      signal_stream(batch.sample_ids, 10, 3, -6, 1)};
  return batch;
}

torch::Tensor reconstruction_loss(TrainingModel& model,
                                   const SyntheticBatch& clean,
                                   const cmt::SpanMaskConfig& config,
                                   int64_t step, uint64_t seed) {
  const double probability = cmt::sample_mask_probability(config, seed, step);
  auto mask = [&](const cmt::StreamBatch& stream, uint64_t stream_key) {
    return cmt::make_span_mask(
        stream.valid, clean.sample_ids, probability,
        config.min_span, config.max_span, seed, step, stream_key);
  };
  const auto target_mask = mask(clean.target, 17);
  const auto target = cmt::mask_span_features(clean.target, target_mask, config);
  std::vector<cmt::StreamBatch> contexts;
  std::vector<torch::Tensor> context_masks;
  for (size_t index = 0; index < clean.contexts.size(); ++index) {
    auto selected = mask(clean.contexts[index], 31 + index);
    contexts.push_back(cmt::mask_span_features(clean.contexts[index], selected, config));
    context_masks.push_back(selected);
  }
  const auto encoded = model->encoder->forward(target, contexts);
  const std::vector<int64_t> reconstructed_columns{0, 1, 2};
  std::vector<torch::Tensor> losses;
  for (size_t index = 0; index < contexts.size(); ++index) {
    losses.push_back(cmt::masked_reconstruction_loss(
        model->reconstruction->forward(encoded.context_memories[index]),
        clean.contexts[index].features, context_masks[index], reconstructed_columns));
  }
  losses.push_back(cmt::masked_reconstruction_loss(
      model->reconstruction->forward(encoded.target_memory), clean.target.features,
      target_mask, reconstructed_columns));
  return cmt::mean_stream_loss(losses);
}

double evaluate(TrainingModel& model, const SyntheticBatch& validation,
                const cmt::SpanMaskConfig& mask_config) {
  torch::NoGradGuard no_grad;
  model->eval();
  // Identical samples and masks at both evaluations; separate from updates.
  return reconstruction_loss(model, validation, mask_config, 0, 73).item<double>();
}

void require_finite_gradients(const TrainingModel& model) {
  for (const auto& parameter : model->parameters()) {
    if (parameter.grad().defined() &&
        !torch::isfinite(parameter.grad()).all().item<bool>()) {
      throw std::runtime_error("non-finite training gradient");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int64_t steps = 120;
    if (argc != 1) {
      if (argc != 3 || std::string(argv[1]) != "--steps") {
        throw std::runtime_error("usage: cmt_train_synthetic [--steps positive_integer]");
      }
      size_t parsed = 0;
      steps = std::stoll(argv[2], &parsed);
      if (parsed != std::string(argv[2]).size() || steps <= 0) {
        throw std::runtime_error("--steps must be a positive integer");
      }
    }
    torch::set_num_threads(1);
    torch::manual_seed(11);
    cmt::EncoderConfig config;
    config.input_dim = 9;
    config.model_dim = 32;
    config.heads = 4;
    config.context_layers = 1;
    config.target_layers = 1;
    config.ffn_multiplier = 2;
    config.dropout = 0.0;
    TrainingModel model(config);
    cmt::SpanMaskConfig mask_config;
    mask_config.probabilities = {0.35, 0.55};
    mask_config.min_span = 1;
    mask_config.max_span = 3;
    mask_config.masked_feature_indices = {0, 1, 2, 3};
    mask_config.indicator_feature_index = 8;

    const auto validation = signals(-10000);
    const double initial = evaluate(model, validation, mask_config);
    const auto encoder_before = model->encoder->input_projection->weight.detach().clone();
    const auto head_before = model->reconstruction->reconstruction->weight.detach().clone();
    torch::optim::AdamW optimizer(
        model->parameters(), torch::optim::AdamWOptions(8e-4).weight_decay(5e-3));
    model->train();
    for (int64_t step = 1; step <= steps; ++step) {
      auto batch = signals((step - 1) * 8);
      const auto loss = reconstruction_loss(model, batch, mask_config, step, 37);
      if (!torch::isfinite(loss).item<bool>()) {
        throw std::runtime_error("non-finite training loss");
      }
      optimizer.zero_grad();
      loss.backward();
      require_finite_gradients(model);
      torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
      optimizer.step();
      if (step == 1 || step % 40 == 0 || step == steps) {
        std::cout << "step=" << step << " training_mse="
                  << loss.item<double>() << '\n';
      }
    }
    const double final = evaluate(model, validation, mask_config);
    const double encoder_change =
        (model->encoder->input_projection->weight - encoder_before).abs().max().item<double>();
    const double head_change =
        (model->reconstruction->reconstruction->weight - head_before).abs().max().item<double>();
    if (!std::isfinite(initial) || !std::isfinite(final) || initial <= 0.0 ||
        !std::isfinite(encoder_change) || !std::isfinite(head_change) ||
        encoder_change <= 0.0 || head_change <= 0.0) {
      throw std::runtime_error("encoder and reconstruction head must both learn finite updates");
    }
    std::cout << std::setprecision(8)
              << "fixed_synthetic_validation_before=" << initial << '\n'
              << "fixed_synthetic_validation_after=" << final << '\n'
              << "validation_ratio=" << final / initial << '\n';
    if (final >= initial * 0.7) {
      throw std::runtime_error(
          "synthetic learning check requires at least 30% error reduction; "
          "try the default step count if you shortened this run");
    }
    std::cout << "Synthetic reconstruction learning check passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
