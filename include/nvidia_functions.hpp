#ifndef NVIDIA_FUNCTIONS_HPP_
#define NVIDIA_FUNCTIONS_HPP_

#include "math_functions.hpp"
#include "tensor.hpp"
#include "sparse_tensor.hpp"
#include "common.hpp"

#define INVERSE 1
#define FORWARD -1

#define SHIFT true
#define NON_SHIFT false

namespace MRI
{

/**
 * @brief A wrapper around several functions from nvidia CUDA library.
 *        Performing CUDA Implementation of numpy.fft.ifftshift(),
 *        numpy.fft.ifftn(), numpy.fft.fftshift(), numpy.fft.fftn(),
 *        scipy.sparse.csr_matrix.dot()
 */

// CUDA Implementation of numpy.fft.ifftshift() - if direction == INVERSE
//                     and numpy.fft.fftshift() - if direction == FORWARD
// Only supports calculating over one single axis
// TODO: Extension - a single axis shift to multi-axes
template <typename Dtype>
void gpu_fftshift(Tensor<Dtype> &tensor, const int &axis, const int &direction);

// CUDA Implementation of numpy.fft.ifftn() - if direction == INVERSE
//                     and numpy.fft.fftn() - if direction == FORWARD
// Supports 1D/2D complex-to-complex transforms for single precision.
template <typename Dtype>
void gpu_fft(Tensor<Dtype> &tensor, const vector<int> &axes,
             const int &direction, const bool &shift = NON_SHIFT);

// CUDA Implementation of scipy.sparse.csr_matrix.dot()
// This function performs the following matrix-matrix operations:
//                          C = A ∗ B
// A is a sparse matrix that is defined in CSR storage format
// B and C are dense matrices

// Attention: B is not const-qualified, because we have to change the data layout
// to match cuSPARSE library
// TODO: try cusparseSpMM instead
template <typename Dtype>
void gpu_dot(const SparseTensor<Dtype> &tensorA, Tensor<Dtype> &tensorB,
             Tensor<Dtype> &tensorC);

} // namespace MRI

#endif // NVIDIA_FUNCTIONS_HPP_