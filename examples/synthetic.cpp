#include <cmt/encoder.hpp>

#include <cmath>
#include <iostream>

namespace {

cmt::StreamBatch make_stream(int64_t length, int64_t stride, int64_t first,
                             int64_t stream_id, int64_t features) {
  const auto times = torch::arange(length, torch::kInt64) * stride + first;
  const auto positions = times.unsqueeze(0).repeat({2, 1});
  const auto frequencies = torch::arange(1, features + 1, torch::kFloat32);
  auto values = torch::sin(positions.to(torch::kFloat32).unsqueeze(-1) *
                           frequencies.view({1, 1, features}) * 0.03);
  values += 0.05 * torch::randn_like(values);

  cmt::StreamBatch stream;
  stream.features = values;
  stream.positions = positions;
  stream.available_at = positions.clone();
  stream.valid = torch::ones({2, length}, torch::kBool);
  stream.stream_id = stream_id;
  stream.scale_id = stream_id;
  return stream;
}

}  // namespace

int main() {
  try {
    torch::manual_seed(7);
    torch::set_num_threads(1);
    cmt::EncoderConfig config;
    cmt::CausalMultiStreamEncoder encoder(config);
    auto target = make_stream(48, 1, 0, 0, config.input_dim);
    auto medium = make_stream(12, 4, 4, 1, config.input_dim);
    auto slow = make_stream(3, 16, 16, 2, config.input_dim);

    const auto output = encoder->forward(target, {slow, medium});
    // A scalar probe verifies autograd; it is not a task-specific objective.
    const auto probe = output.target_memory.select(-1, 0).square().mean();
    probe.backward();
    double gradient_energy = 0.0;
    for (const auto& parameter : encoder->parameters()) {
      if (parameter.grad().defined()) {
        gradient_energy += parameter.grad().square().sum().item<double>();
      }
    }
    if (!torch::isfinite(output.target_memory).all().item<bool>() ||
        !std::isfinite(gradient_energy) || gradient_energy <= 0.0) {
      throw std::runtime_error("synthetic forward/backward check failed");
    }
    std::cout << "Target memory: " << output.target_memory.sizes() << '\n'
              << "Context memory: " << output.context_memory.sizes() << '\n'
              << "Synthetic forward/backward check passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
