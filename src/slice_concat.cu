#include <cuda_runtime.h>
#include <algorithm>

#include "slice_concat.hpp"

namespace MRI
{

template <typename Dtype>
__global__ void slice_kernel(const int n, const Dtype *in_data,
                             const int num_slices, const int slice_size,
                             const int input_slice_axis,
                             const int offset_slice_axis, Dtype *out_data)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int slice_num = index / slice_size;
        int slice_index = index % slice_size;
        int input_index = slice_index +
                          (slice_num * input_slice_axis + offset_slice_axis) * slice_size;
        out_data[index] = in_data[input_index];
    }
}

template <typename Dtype>
void gpu_slice(const Tensor<Dtype> &input, vector<shared_ptr<Tensor<Dtype>>> &output,
               const int &axis, const bool &copy)
{
    int axis_ = input.CanonicalAxisIndex(axis);
    output.clear();
    vector<int> shape = input.shape();
    int size = shape[axis_];
    if (size == 1)
    {
        output.push_back(shared_ptr<Tensor<Dtype>>(new Tensor<Dtype>(shape)));
        if (copy)
        {
            output[0]->copyfrom(input);
        }
        else
        {
            output[0]->sharedata(input);
        }
        return;
    }
    shape[axis_] = 1;
    for (int i = 0; i < size; ++i)
    {
        output.push_back(shared_ptr<Tensor<Dtype>>(new Tensor<Dtype>(shape)));
    }

    const int slice_size_ = input.count(axis_ + 1);
    if (axis_ == 0)
    {
        int offset = 0;
        for (int i = 0; i < size; ++i)
        {
            // TODO: it's not safe. Re-implement this
            // Since c++20 shared_ptr owns the aliasing constructor
            output[i]->set_gpu_data((Dtype *)input.gpu_data() + offset);
            offset += slice_size_;
        }
    }
    else
    {
        int offset_slice_axis = 0;
        const Dtype *input_data = input.gpu_data();
        const int input_slice_axis = size;
        const int num_slices_ = input.count(0, axis_);
        const int nthreads = slice_size_ * num_slices_;
        for (int i = 0; i < size; ++i)
        {
            Dtype *output_data = output[i]->mutable_gpu_data();
            slice_kernel<Dtype><<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
                nthreads, input_data, num_slices_, slice_size_,
                input_slice_axis, offset_slice_axis, output_data);
            ++offset_slice_axis;
        }
    }
}

// Instantiate the function
template void gpu_slice<float>(const Tensor<float> &input,
                               vector<shared_ptr<Tensor<float>>> &output,
                               const int &axis, const bool &copy);
template void gpu_slice<cuFloatComplex>(const Tensor<cuFloatComplex> &input,
                                        vector<shared_ptr<Tensor<cuFloatComplex>>> &output,
                                        const int &axis, const bool &copy);

template <typename Dtype>
__global__ void slice_kernel(const int n, const Dtype *in_data,
                             const int num_slices, const int slice_size,
                             const int input_slice_axis, const int output_slice_axis,
                             const int offset_slice_axis, Dtype *out_data)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int total_slice_size = slice_size * output_slice_axis;
        const int slice_num = index / total_slice_size;
        const int slice_index = index % total_slice_size;
        const int input_index = slice_index +
                                (slice_num * input_slice_axis + offset_slice_axis) * slice_size;
        out_data[index] = in_data[input_index];
    }
}

template <typename Dtype>
void gpu_slice(const Tensor<Dtype> &input,
               vector<shared_ptr<Tensor<Dtype>>> &output,
               const int &axis, const vector<int> &slice_points)
{
    int axis_ = input.CanonicalAxisIndex(axis);
    output.clear();
    if (input.shape(axis_) == 1)
    {
        output.push_back(shared_ptr<Tensor<Dtype>>(new Tensor<Dtype>(input.shape())));
        output[0]->copyfrom(input);
        return;
    }

    int size = slice_points.size();
    vector<int> shape = input.shape();
    for (int i = 0; i < size; ++i)
    {
        if (i == 0)
        {
            shape[axis_] = slice_points[0];
        }
        else
        {
            shape[axis_] = slice_points[i] - slice_points[i - 1];
        }
        output.push_back(shared_ptr<Tensor<Dtype>>(new Tensor<Dtype>(shape)));
    }
    const int slice_size_ = input.count(axis_ + 1);
    if (axis_ == 0)
    {
        int offset = 0;
        for (int i = 0; i < size; ++i)
        {
            // TODO: it's not safe. Re-implement this
            // Since c++20 shared_ptr owns the aliasing constructor
            output[i]->set_gpu_data((Dtype *)input.gpu_data() + offset);
            offset += output[i]->shape(0) * slice_size_;
        }
    }
    else
    {
        int offset_slice_axis = 0;
        const Dtype *input_data = input.gpu_data();
        const int input_slice_axis = input.shape(axis_);
        const int num_slices_ = input.count(0, axis_);
        for (int i = 0; i < size; ++i)
        {
            Dtype *output_data = output[i]->mutable_gpu_data();
            const int output_slice_axis = output[i]->shape(axis_);
            const int output_slice_size = output_slice_axis * slice_size_;
            const int nthreads = output_slice_size * num_slices_;
            slice_kernel<Dtype><<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
                nthreads, input_data, num_slices_, slice_size_,
                input_slice_axis, output_slice_axis, offset_slice_axis, output_data);
            offset_slice_axis += output_slice_axis;
        }
    }
}

// Instantiate the function
template void gpu_slice<float>(const Tensor<float> &input,
                               vector<shared_ptr<Tensor<float>>> &output,
                               const int &axis, const vector<int> &slice_points);
template void gpu_slice<cuFloatComplex>(const Tensor<cuFloatComplex> &input,
                                        vector<shared_ptr<Tensor<cuFloatComplex>>> &output,
                                        const int &axis, const vector<int> &slice_points);

template <typename Dtype>
__global__ void concat_kernel(const int n, const Dtype *in_data,
                              const int num_concats, const int concat_size,
                              const int output_concat_axis, const int input_concat_axis,
                              const int offset_concat_axis, Dtype *out_data)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int total_concat_size = concat_size * input_concat_axis;
        const int concat_num = index / total_concat_size;
        const int concat_index = index % total_concat_size;
        const int output_index = concat_index +
                                 (concat_num * output_concat_axis + offset_concat_axis) * concat_size;
        out_data[output_index] = in_data[index];
    }
}

template <typename Dtype>
void gpu_concat(const vector<shared_ptr<Tensor<Dtype>>> &input,
                Tensor<Dtype> &output, const int &axis, const bool &copy)
{
    if (input.size() == 1)
    {
        if (copy)
        {
            output.copyfrom(*(input[0]));
        }
        else
        {
            output.sharedata(*(input[0]));
        }
        return;
    }

    vector<int> shape = input[0]->shape();
    int sum = 0;
    int axis_ = input[0]->CanonicalAxisIndex(axis);
    for (int i = 0; i < input.size(); ++i)
    {
        sum += input[i]->shape(axis_);
    }
    shape[axis_] = sum;
    output.reshape(shape);

    Dtype *output_data = output.mutable_gpu_data();
    int offset_concat_axis = 0;
    const int output_concat_axis = output.shape(axis_);
    const int concat_input_size_ = input[0]->count(axis_ + 1);
    const int num_concats_ = input[0]->count(0, axis_);
    for (int i = 0; i < input.size(); ++i)
    {
        const Dtype *input_data = input[i]->gpu_data();
        const int input_concat_axis = input[i]->shape(axis_);
        const int input_concat_size = input_concat_axis * concat_input_size_;
        const int nthreads = input_concat_size * num_concats_;
        concat_kernel<Dtype><<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
            nthreads, input_data, num_concats_, concat_input_size_,
            output_concat_axis, input_concat_axis, offset_concat_axis, output_data);
        offset_concat_axis += input_concat_axis;
    }
}

// Instantiate the function
template void gpu_concat<float>(const vector<shared_ptr<Tensor<float>>> &input,
                                Tensor<float> &output, const int &axis, const bool &copy);
template void gpu_concat<cuFloatComplex>(const vector<shared_ptr<Tensor<cuFloatComplex>>> &input,
                                         Tensor<cuFloatComplex> &output, const int &axis,
                                         const bool &copy);

__device__ int compute_uncropped_index(
    int index,
    const int ndims,
    const int *in_strides,
    const int *out_strides)
{
    int out_index = index;
    int in_index = 0;
    for (int i = 0; i < ndims; ++i)
    {
        int coord = out_index / out_strides[i];
        out_index -= coord * out_strides[i];
        in_index += in_strides[i] * coord;
    }
    return in_index;
}

template <typename Dtype>
__global__ void crop_kernel(const int nthreads,
                            const int ndims,
                            const int *in_strides,
                            const int *out_strides,
                            const Dtype *in, Dtype *out)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int in_index = compute_uncropped_index(
            index, ndims, in_strides, out_strides);
        out[index] = in[in_index];
    }
}

template <typename Dtype>
void gpu_crop(const Tensor<Dtype> &input, Tensor<Dtype> &output,
              const vector<int> &offset, const int &axis)
{
    int axis_ = input.CanonicalAxisIndex(axis);
    vector<int> shape_ = input.shape();
    for (int i = axis_, j = 0; i < min(shape_.size(), axis_ + offset.size()); ++i, ++j)
    {
        shape_[i] = offset[j];
    }
    Tensor<Dtype> tmp(shape_);

    // Compute strides
    int num_axes = input.num_axes();
    vector<int> in_strides_(num_axes);
    vector<int> out_strides_(num_axes);
    for (int i = 0; i < num_axes; ++i)
    {
        in_strides_[i] = input.count(i + 1, num_axes);
        out_strides_[i] = tmp.count(i + 1, num_axes);
    }

    // Move the parameters to the device
    int *in_strides, *out_strides;
    size_t size = num_axes * sizeof(int);
    CUDA_CHECK(cudaMalloc((void **)&in_strides, size));
    CUDA_CHECK(cudaMalloc((void **)&out_strides, size));
    CUDA_CHECK(cudaMemcpy(in_strides, &(*in_strides_.begin()), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_strides, &(*out_strides_.begin()), size, cudaMemcpyHostToDevice));

    const Dtype *in_data = input.gpu_data();
    Dtype *out_data = tmp.mutable_gpu_data();
    int nthreads = tmp.count();
    crop_kernel<<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
        nthreads,
        num_axes,
        in_strides,
        out_strides,
        in_data, out_data);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaFree(in_strides));
    CUDA_CHECK(cudaFree(out_strides));
    output.sharedata(tmp);
}

// Instantiate the function
template void gpu_crop<float>(const Tensor<float> &input, Tensor<float> &output,
                              const vector<int> &offset, const int &axis);
template void gpu_crop<cuFloatComplex>(const Tensor<cuFloatComplex> &input,
                                       Tensor<cuFloatComplex> &output,
                                       const vector<int> &offset, const int &axis);

__device__ int compute_uncropped_index(
    int index,
    const int ndims,
    const int *in_strides,
    const int *out_strides,
    const int *offsets)
{
    int out_index = index;
    int in_index = 0;
    for (int i = 0; i < ndims; ++i)
    {
        int coord = out_index / out_strides[i];
        out_index -= coord * out_strides[i];
        in_index += in_strides[i] * (coord + offsets[i]);
    }
    return in_index;
}

template <typename Dtype>
__global__ void crop_kernel(const int nthreads,
                            const int ndims,
                            const int *in_strides,
                            const int *out_strides,
                            const int *offsets,
                            const Dtype *in, Dtype *out)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int in_index = compute_uncropped_index(
            index, ndims, in_strides, out_strides, offsets);
        out[index] = in[in_index];
    }
}

template <typename Dtype>
void gpu_crop(const Tensor<Dtype> &input, Tensor<Dtype> &output,
              const vector<int> &start_points, const vector<int> &offset, const int &axis)
{
    int axis_ = input.CanonicalAxisIndex(axis);
    vector<int> shape_ = input.shape();
    for (int i = axis_, j = 0; i < min(shape_.size(), axis_ + offset.size()); ++i, ++j)
    {
        shape_[i] = offset[j];
    }
    Tensor<Dtype> tmp(shape_);

    // Compute strides
    int num_axes = input.num_axes();
    vector<int> in_strides_(num_axes);
    vector<int> out_strides_(num_axes);
    vector<int> offsets_(num_axes);
    for (int i = 0; i < num_axes; ++i)
    {
        in_strides_[i] = input.count(i + 1, num_axes);
        out_strides_[i] = tmp.count(i + 1, num_axes);
    }

    // Determine crop offsets and the new shape post-crop.
    for (int i = 0; i < num_axes; ++i)
    {
        int crop_offset = 0;
        if (i >= axis_)
        {
            crop_offset = start_points[i - axis_];
        }
        offsets_[i] = crop_offset;
    }

    // Move the parameters to the device
    int *in_strides, *out_strides, *offsets;
    size_t size = num_axes * sizeof(int);
    CUDA_CHECK(cudaMalloc((void **)&in_strides, size));
    CUDA_CHECK(cudaMalloc((void **)&out_strides, size));
    CUDA_CHECK(cudaMalloc((void **)&offsets, size));
    CUDA_CHECK(cudaMemcpy(in_strides, &(*in_strides_.begin()), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_strides, &(*out_strides_.begin()), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(offsets, &(*offsets_.begin()), size, cudaMemcpyHostToDevice));

    const Dtype *in_data = input.gpu_data();
    Dtype *out_data = tmp.mutable_gpu_data();
    int nthreads = tmp.count();
    crop_kernel<<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
        nthreads,
        num_axes,
        in_strides,
        out_strides,
        offsets,
        in_data, out_data);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaFree(in_strides));
    CUDA_CHECK(cudaFree(out_strides));
    CUDA_CHECK(cudaFree(offsets));
    output.sharedata(tmp);
}

// Instantiate the function
template void gpu_crop<float>(const Tensor<float> &input, Tensor<float> &output,
                              const vector<int> &start_points, const vector<int> &offset,
                              const int &axis);
template void gpu_crop<cuFloatComplex>(const Tensor<cuFloatComplex> &input,
                                       Tensor<cuFloatComplex> &output,
                                       const vector<int> &start_points,
                                       const vector<int> &offset, const int &axis);

} // namespace MRI
