#ifndef PERMUTE_HPP_
#define PERMUTE_HPP_

#include <vector>

#include "tensor.hpp"

namespace MRI
{

/**
 * @brief Permute the input Tensor by changing the memory order of the data.
 *
 */

template <typename Dtype>
void transpose(Tensor<Dtype> &tensor, const vector<int> &permute_order_);

template <typename Dtype>
void swapaxes(Tensor<Dtype> &tensor, const int &axis1, const int &axis2);

} // namespace MRI

#endif // PERMUTE_HPP_