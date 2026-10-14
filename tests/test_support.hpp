#pragma once

#include <torch/torch.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cmt_test {

inline void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

inline double max_abs_diff(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs) {
  require(lhs.defined() && rhs.defined(), "comparison tensors must be defined");
  require(lhs.sizes() == rhs.sizes(), "comparison tensors must have equal shapes");
  if (lhs.numel() == 0) {
    return 0.0;
  }
  return (lhs.detach().to(torch::kFloat64) - rhs.detach().to(torch::kFloat64))
      .abs()
      .max()
      .item<double>();
}

inline void require_allclose(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    double rtol,
    double atol,
    const std::string& message) {
  if (!torch::allclose(lhs, rhs, rtol, atol)) {
    std::ostringstream detail;
    detail << message << " (max abs diff " << max_abs_diff(lhs, rhs) << ")";
    throw std::runtime_error(detail.str());
  }
}

inline void require_finite(
    const torch::Tensor& tensor,
    const std::string& message) {
  require(tensor.defined(), message + " (undefined tensor)");
  require(torch::isfinite(tensor).all().item<bool>(), message);
}

inline void require_zero(
    const torch::Tensor& tensor,
    double tolerance,
    const std::string& message) {
  require(tensor.defined(), message + " (undefined tensor)");
  require_finite(tensor, message + " (non-finite tensor)");
  const double maximum = tensor.numel() == 0
      ? 0.0
      : tensor.detach().to(torch::kFloat64).abs().max().item<double>();
  if (maximum > tolerance) {
    std::ostringstream detail;
    detail << message << " (max abs value " << maximum << ")";
    throw std::runtime_error(detail.str());
  }
}

inline void require_zero_or_undefined_gradient(
    const torch::Tensor& gradient,
    double tolerance,
    const std::string& message) {
  if (gradient.defined()) {
    require_zero(gradient, tolerance, message);
  }
}

template <typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
  bool threw = false;
  try {
    callable();
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, message);
}

inline void run_case(
    const std::string& name,
    const std::function<void()>& body,
    int& failures) {
  try {
    body();
    std::cout << "[PASS] " << name << '\n';
  } catch (const std::exception& error) {
    ++failures;
    std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
  }
}

}  // namespace cmt_test
