#ifndef MATH_FUNCTIONS_HPP_
#define MATH_FUNCTIONS_HPP_

#include "tensor.hpp"
#include "common.hpp"

namespace MRI
{

// Set all the elements of tensor as alpha
template <typename Dtype>
void gpu_set(Tensor<Dtype> &tensor, const Dtype &alpha);

// Element-wise addition : C = A + B
template <typename Dtype>
void gpu_add(const Tensor<Dtype> &tensorA,
             const Tensor<Dtype> &tensorB, Tensor<Dtype> &tensorC);

// Element-wise multiply : C = A * B
template <typename Dtype>
void gpu_mul(const Tensor<Dtype> &tensorA,
             const Tensor<Dtype> &tensorB, Tensor<Dtype> &tensorC);

// Normalization for cuFFT
template <typename Dtype>
void gpu_normalizing(Tensor<Dtype> &tensor, const int &scale);

// Reduce a tensor to another that keeps the sum along a dimension
template <typename Dtype>
void gpu_reduce(Tensor<Dtype> &input,
                Tensor<Dtype> &output, const int &axis);

} // namespace MRI

#endif // MATH_FUNCTIONS_HPP_