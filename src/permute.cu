#include <algorithm>
#include <vector>

#include "permute.hpp"

namespace MRI
{

// The kernel which does the permute.
// This kernel can be used to perform CUDA Implementation of numpy.transpose() and numpy.swapaxes()
template <typename Dtype>
__global__ void PermuteKernel(const int n,
                              const Dtype * in_data, const int *permute_order,
                              const int *old_steps, const int *new_steps, const int num_axes,
                              Dtype *const out_data)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int temp_idx = index;
        int old_idx = 0;
        for (int i = 0; i < num_axes; ++i)
        {
            int order = permute_order[i];
            old_idx += (temp_idx / new_steps[i]) * old_steps[order];
            temp_idx %= new_steps[i];
        }
        out_data[index] = in_data[old_idx];
    }
}

template <typename Dtype>
void transpose(Tensor<Dtype> &tensor, const vector<int> &permute_order_)
{
    int num_axes = tensor.num_axes();
    CHECK_EQ(num_axes, permute_order_.size());
    const Dtype *in_data = tensor.gpu_data();
    int count = tensor.count();

    int new_steps_[num_axes], old_steps_[num_axes];
    // Used to record the total number of elements in a few dimensions before the replacement
    old_steps_[num_axes - 1] = 1;
    for (int i = num_axes - 2; i >= 0; --i)
    {
        old_steps_[i] = old_steps_[i + 1] * tensor.shape(i + 1);
    }

    vector<int> new_shape = tensor.shape();
    for (int i = 0; i < num_axes; ++i)
    {
        new_shape[i] = tensor.shape(permute_order_[i]);
    }
    tensor.reshape(new_shape);

    // Used to record the total number of elements in a few dimensions after the replacement
    new_steps_[num_axes - 1] = 1;
    for (int i = num_axes - 2; i >= 0; --i)
    {
        new_steps_[i] = new_steps_[i + 1] * tensor.shape(i + 1);
    }

    // Move the parameters to the device
    int *permute_order, *new_steps, *old_steps;
    size_t size = num_axes * sizeof(int);
    CUDA_CHECK(cudaMalloc((void **)&permute_order, size));
    CUDA_CHECK(cudaMalloc((void **)&new_steps, size));
    CUDA_CHECK(cudaMalloc((void **)&old_steps, size));
    CUDA_CHECK(cudaMemcpy(permute_order, &(*permute_order_.begin()), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(new_steps, new_steps_, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(old_steps, old_steps_, size, cudaMemcpyHostToDevice));
    
    Tensor<Dtype> tmp;
    tmp.reshapelike(tensor);
    Dtype *out_data = tmp.mutable_gpu_data();

    PermuteKernel<Dtype><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
        count, in_data, permute_order, old_steps, new_steps,
        num_axes, out_data);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaFree(permute_order));
    CUDA_CHECK(cudaFree(new_steps));
    CUDA_CHECK(cudaFree(old_steps));
    tensor.sharedata(tmp);
}

// Instantiate the function
template void transpose<float>(Tensor<float> &tensor, const vector<int> &permute_order_);
template void transpose<cuFloatComplex>(Tensor<cuFloatComplex> &tensor, const vector<int> &permute_order_);

template <typename Dtype>
void swapaxes(Tensor<Dtype> &tensor, const int &axis1, const int &axis2)
{
    int axis1_ = tensor.CanonicalAxisIndex(axis1);
    int axis2_ = tensor.CanonicalAxisIndex(axis2);
    int num_axes = tensor.num_axes();
    vector<int> permute_order(num_axes);

    for (int i = 0; i < num_axes; ++i)
    {
        permute_order[i] = i;
    }
    permute_order[axis1_] = axis2_;
    permute_order[axis2_] = axis1_;

    transpose(tensor, permute_order);
}

// Instantiate the function
template void swapaxes<float>(Tensor<float> &tensor, const int &axis1, const int &axis2);
template void swapaxes<cuFloatComplex>(Tensor<cuFloatComplex> &tensor, const int &axis1, const int &axis2);

} // namespace MRI