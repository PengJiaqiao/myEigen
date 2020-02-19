#ifndef SLICE_CONCAT_HPP_
#define SLICE_CONCAT_HPP_

#include "tensor.hpp"
#include "common.hpp"

namespace MRI
{

// Split input tensor into several output tensors along a specified dimension
// output.shape(axis) = 1
// @Param: axis: the axis to do the slice along, can be -1 for the last axis
//
template <typename Dtype>
void gpu_slice(const Tensor<Dtype> &input, vector<shared_ptr<Tensor<Dtype>>> &output,
               const int &axis, const bool &copy = false);

// Split input tensor into several output tensors along a specified dimension
// @Param: axis: the axis to do the slice along, can be -1 for the last axis
//         slice_points: the end point of each slice
template <typename Dtype>
void gpu_slice(const Tensor<Dtype> &input,
               vector<shared_ptr<Tensor<Dtype>>> &output,
               const int &axis, const vector<int> &slice_points);

// Concatenate input tensors into a single output tensor
// @Param: axis: the axis to do the concatenation along, can be -1 for the last axis
//
template <typename Dtype>
void gpu_concat(const vector<shared_ptr<Tensor<Dtype>>> &input,
                Tensor<Dtype> &output, const int &axis, const bool &copy = false);

// Crop input tensor with offset from the axis
// @Param: offset: crop [tensor.shape(0), ..., tensor.shape(axis), ...] to
//         [..., offset[0], ...], and so on.
//
template <typename Dtype>
void gpu_crop(const Tensor<Dtype> &input, Tensor<Dtype> &output,
              const vector<int> &offset, const int &axis = 0);

// Crop input tensor with offset from the axis
// @Param: start_points & offset: crop [tensor.shape(0), ..., tensor.shape(axis), ...] to
//         [..., start_points : start_points + offset[0], ...], and so on.
//
template <typename Dtype>
void gpu_crop(const Tensor<Dtype> &input, Tensor<Dtype> &output,
              const vector<int> &start_points, const vector<int> &offset, const int &axis = 0);

} // namespace MRI

#endif // SLICE_CONCAT_HPP_