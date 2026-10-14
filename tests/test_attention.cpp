#include <cmt/encoder.hpp>

#include "test_support.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace {

void causal_mask_uses_availability_time() {
  const auto q_available = torch::tensor(
      {{4, 9}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto k_available = torch::tensor(
      {{8, 2, 9}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto q_valid = torch::tensor(
      {{true, true}}, torch::TensorOptions().dtype(torch::kBool));
  const auto k_valid = torch::tensor(
      {{true, true, true}}, torch::TensorOptions().dtype(torch::kBool));

  const auto actual = cmt::make_causal_mask(
      q_available, k_available, q_valid, k_valid);
  const auto expected = torch::tensor(
      {{{false, true, false}, {true, true, true}}},
      torch::TensorOptions().dtype(torch::kBool));

  cmt_test::require(actual.scalar_type() == torch::kBool,
                    "causal mask must be boolean");
  cmt_test::require(actual.equal(expected),
                    "causal mask must compare availability timestamps, including equality");

  auto invalid_key = k_valid.clone();
  invalid_key[0][1].fill_(false);
  auto masked_key = cmt::make_causal_mask(
      q_available, k_available, q_valid, invalid_key);
  cmt_test::require(!masked_key[0][0][1].item<bool>(),
                    "invalid keys must never be visible");

  auto invalid_query = q_valid.clone();
  invalid_query[0][1].fill_(false);
  auto masked_query = cmt::make_causal_mask(
      q_available, k_available, invalid_query, k_valid);
  cmt_test::require(!masked_query[0][1].any().item<bool>(),
                    "invalid queries must have no visible keys");

  const int64_t large = int64_t{1} << 60;
  const auto large_queries = torch::tensor(
      {{large + 1, large + 2}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto large_keys = torch::tensor(
      {{large, large + 1, large + 2}},
      torch::TensorOptions().dtype(torch::kInt64));
  const auto large_actual = cmt::make_causal_mask(
      large_queries, large_keys, q_valid, k_valid);
  const auto large_expected = torch::tensor(
      {{{true, true, false}, {true, true, true}}},
      torch::TensorOptions().dtype(torch::kBool));
  cmt_test::require(
      large_actual.equal(large_expected),
      "large adjacent int64 timestamps must remain exactly distinguishable");

  cmt_test::require_throws(
      [&] {
        cmt::make_causal_mask(
            large_queries, large_keys.to(torch::kInt32), q_valid, k_valid);
      },
      "mixed int64/int32 availability timestamps must be rejected");
  cmt_test::require_throws(
      [&] {
        cmt::make_causal_mask(
            large_queries.to(torch::kFloat64), large_keys, q_valid, k_valid);
      },
      "floating-point availability timestamps must be rejected");
}

void all_blocked_attention_is_exactly_inert() {
  torch::manual_seed(101);
  cmt::RotaryAttention attention(16, 4, 0.0, 10000.0);
  attention->eval();

  auto query = torch::randn({2, 3, 16}).set_requires_grad(true);
  auto key_value = torch::randn({2, 5, 16}).set_requires_grad(true);
  const auto q_position = torch::tensor(
      {{0, 1, 2}, {7, 8, 9}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto k_position = torch::tensor(
      {{-4, -3, -2, -1, 0}, {2, 3, 4, 5, 6}},
      torch::TensorOptions().dtype(torch::kInt64));
  const auto allow = torch::zeros({2, 3, 5}, torch::kBool);

  const auto output = attention->forward(
      query, key_value, q_position, k_position, allow);
  cmt_test::require_zero(
      output, 0.0, "an all-blocked row must emit exactly zero, including output bias");

  output.sum().backward();
  cmt_test::require_zero_or_undefined_gradient(
      query.grad(), 0.0, "all-blocked attention produced query gradients");
  cmt_test::require_zero_or_undefined_gradient(
      key_value.grad(), 0.0, "all-blocked attention produced key/value gradients");
  for (const auto& parameter : attention->named_parameters()) {
    cmt_test::require_zero_or_undefined_gradient(
        parameter.value().grad(), 0.0,
        "all-blocked attention produced parameter gradients for " + parameter.key());
  }
}

void rotary_positions_have_relative_shift_symmetry() {
  torch::manual_seed(202);
  cmt::RotaryAttention attention(24, 4, 0.0, 10000.0);
  attention->eval();

  const auto query = torch::randn({1, 3, 24});
  const auto key_value = torch::randn({1, 4, 24});
  const auto q_position = torch::tensor(
      {{255, 256, 261}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto k_position = torch::tensor(
      {{250, 256, 259, 265}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto allow = torch::ones({1, 3, 4}, torch::kBool);

  const auto baseline = attention->forward(
      query, key_value, q_position, k_position, allow);
  const auto shifted = attention->forward(
      query, key_value, q_position + 137, k_position + 137, allow);
  cmt_test::require_allclose(
      baseline, shifted, 2e-5, 2e-6,
      "RoPE attention must be approximately invariant to a common position shift");

  auto adjacent_position = q_position.clone();
  adjacent_position[0][1].add_(1);
  const auto adjacent = attention->forward(
      query, key_value, adjacent_position, k_position, allow);
  cmt_test::require(
      cmt_test::max_abs_diff(baseline, adjacent) > 1e-7,
      "adjacent positions must remain distinguishable");
}

void low_precision_positions_are_rejected() {
  torch::manual_seed(303);
  cmt::RotaryAttention attention(16, 4, 0.0, 10000.0);
  attention->eval();
  const auto query = torch::randn({1, 2, 16});
  const auto key_value = torch::randn({1, 3, 16});
  const auto positions_q = torch::tensor(
      {{0, 1}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto positions_k = torch::tensor(
      {{0, 1, 2}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto allow = torch::ones({1, 2, 3}, torch::kBool);

  cmt_test::require_throws(
      [&] {
        attention->forward(
            query, key_value, positions_q.to(torch::kFloat16), positions_k, allow);
      },
      "float16 query positions must be rejected");
  cmt_test::require_throws(
      [&] {
        attention->forward(
            query, key_value, positions_q, positions_k.to(torch::kBFloat16), allow);
      },
      "bfloat16 key positions must be rejected");
}

void cpu_bfloat16_forward_and_backward_are_finite() {
  torch::manual_seed(304);
  cmt::RotaryAttention attention(16, 4, 0.0, 10000.0);
  attention->to(torch::Device(torch::kCPU), torch::kBFloat16);
  attention->eval();

  auto query = torch::randn({1, 2, 16})
      .to(torch::kBFloat16)
      .set_requires_grad(true);
  auto key_value = torch::randn({1, 3, 16})
      .to(torch::kBFloat16)
      .set_requires_grad(true);
  const auto q_position = torch::tensor(
      {{256, 257}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto k_position = torch::tensor(
      {{255, 256, 257}}, torch::TensorOptions().dtype(torch::kInt64));
  const auto allow = torch::tensor(
      {{{false, false, false}, {true, true, true}}},
      torch::TensorOptions().dtype(torch::kBool));

  const auto output = attention->forward(
      query, key_value, q_position, k_position, allow);
  cmt_test::require(
      output.scalar_type() == torch::kBFloat16,
      "CPU bfloat16 attention must preserve the activation dtype");
  cmt_test::require_finite(
      output, "CPU bfloat16 attention output must be finite");
  cmt_test::require_zero(
      output.select(1, 0), 0.0,
      "a fully blocked CPU bfloat16 query row must emit exactly zero");

  output.select(1, 1).to(torch::kFloat32).square().sum().backward();
  cmt_test::require_finite(
      query.grad(), "CPU bfloat16 query gradients must be finite");
  cmt_test::require_finite(
      key_value.grad(), "CPU bfloat16 key/value gradients must be finite");
  cmt_test::require_zero(
      query.grad().select(1, 0), 0.0,
      "a fully blocked CPU bfloat16 query row must receive zero gradient");
  for (const auto& parameter : attention->named_parameters()) {
    cmt_test::require(
        parameter.value().grad().defined(),
        "CPU bfloat16 parameter gradient is undefined for " + parameter.key());
    cmt_test::require_finite(
        parameter.value().grad(),
        "CPU bfloat16 parameter gradient is non-finite for " + parameter.key());
  }
}

void malformed_attention_inputs_fail_fast() {
  cmt::RotaryAttention attention(16, 4, 0.0, 10000.0);
  const auto query = torch::randn({1, 2, 16});
  const auto key_value = torch::randn({1, 3, 16});
  const auto q_position = torch::zeros({1, 2}, torch::kInt64);
  const auto k_position = torch::zeros({1, 3}, torch::kInt64);

  cmt_test::require_throws(
      [&] {
        attention->forward(
            query, key_value, q_position, k_position,
            torch::ones({1, 2, 2}, torch::kBool));
      },
      "a mask with the wrong key length must be rejected");
  cmt_test::require_throws(
      [&] {
        cmt::RotaryAttention invalid(15, 4, 0.0, 10000.0);
      },
      "model width must be divisible by the head count");
}

}  // namespace

int main() {
  int failures = 0;
  cmt_test::run_case(
      "causal mask uses availability time",
      causal_mask_uses_availability_time,
      failures);
  cmt_test::run_case(
      "all-blocked attention is exactly inert",
      all_blocked_attention_is_exactly_inert,
      failures);
  cmt_test::run_case(
      "rotary relative shift symmetry",
      rotary_positions_have_relative_shift_symmetry,
      failures);
  cmt_test::run_case(
      "low-precision positions rejected",
      low_precision_positions_are_rejected,
      failures);
  cmt_test::run_case(
      "CPU bfloat16 forward and backward",
      cpu_bfloat16_forward_and_backward_are_finite,
      failures);
  cmt_test::run_case(
      "malformed attention inputs fail fast",
      malformed_attention_inputs_fail_fast,
      failures);
  return failures == 0 ? 0 : 1;
}
