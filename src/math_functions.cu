#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <time.h>

#include "math_functions.hpp"

namespace MRI
{

// Operator overload
inline bool operator==(const cuFloatComplex &A, const int &B)
{
    return A.x == 0 && A.y == 0 && B == 0; // For line 74
}

__device__ __forceinline__ cuFloatComplex
operator+(const cuFloatComplex &A, const cuFloatComplex &B)
{
    return make_cuFloatComplex(A.x + B.x, A.y + B.y);
}

__device__ __forceinline__ void operator+=(cuFloatComplex &A, const cuFloatComplex &B)
{
    A.x += B.x;
    A.y += B.y;
}

__device__ __forceinline__ cuFloatComplex
operator*(const cuFloatComplex &A, const cuFloatComplex &B)
{
    return make_cuFloatComplex(A.x * B.x - A.y * B.y,
                               A.y * B.x + A.x * B.y);
}

__device__ __forceinline__ void operator*=(cuFloatComplex &A, const cuFloatComplex &B)
{
    A.x = A.x * B.x - A.y * B.y;
    A.y = A.y * B.x + A.x * B.y;
}

__device__ __forceinline__ cuFloatComplex
operator*(const cuFloatComplex &value, const int &scale)
{
    return make_cuFloatComplex(value.x / scale,
                               value.y / scale);
}

__device__ __forceinline__ void operator/=(cuFloatComplex &value, const int &scale)
{
    value.x /= scale;
    value.y /= scale;
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype *y)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        y[index] = alpha;
    }
}

template <typename Dtype>
void gpu_set(Tensor<Dtype> &tensor, const Dtype &alpha)
{
    int N = tensor.count();
    Dtype *data = tensor.mutable_gpu_data();
    if (alpha == 0)
    {
        CUDA_CHECK(cudaMemset(data, 0, sizeof(Dtype) * N));
        return;
    }

    set_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
        N, alpha, data);
}

// Instantiate the function
template void gpu_set<float>(Tensor<float> &tensor, const float &alpha);
template void gpu_set<cuFloatComplex>(Tensor<cuFloatComplex> &tensor, const cuFloatComplex &alpha);

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype *a,
                           const Dtype *b, Dtype *c)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        c[index] = a[index] + b[index];
    }
}

template <typename Dtype>
__global__ void add_with_stride_kernel(const int n, const Dtype *a,
                                       const Dtype *b, Dtype *c, const int stride)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        c[index] = a[index] + b[index / stride];
    }
}

template <typename Dtype>
void gpu_add(const Tensor<Dtype> &tensorA,
             const Tensor<Dtype> &tensorB, Tensor<Dtype> &tensorC)
{
    if (tensorA.count() != tensorB.count())
    {
        if (tensorA.count() > tensorB.count())
        {
            CHECK_EQ(tensorA.count() % tensorB.count(), 0);
            if (tensorA.count() != tensorC.count() || tensorA.shape() != tensorC.shape())
            {
                LOG(FATAL) << "Trying to element-wise add Tensors to different size.";
            }
            int N = tensorA.count();
            add_with_stride_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
                N, tensorA.gpu_data(), tensorB.gpu_data(), tensorC.mutable_gpu_data(),
                tensorA.count() / tensorB.count());
        }
        else
        {
            CHECK_EQ(tensorB.count() % tensorA.count(), 0);
            if (tensorB.count() != tensorC.count() || tensorB.shape() != tensorC.shape())
            {
                LOG(FATAL) << "Trying to element-wise add Tensors to different size.";
            }
            int N = tensorB.count();
            add_with_stride_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
                N, tensorB.gpu_data(), tensorA.gpu_data(), tensorC.mutable_gpu_data(),
                tensorB.count() / tensorA.count());
        }
    }
    else
    {
        CHECK_EQ(tensorA.count(), tensorC.count());
        int N = tensorA.count();
        add_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
            N, tensorA.gpu_data(), tensorB.gpu_data(), tensorC.mutable_gpu_data());
    }
}

// Instantiate the function
template void gpu_add<float>(const Tensor<float> &tensorA,
                             const Tensor<float> &tensorB, Tensor<float> &tensorC);
template void gpu_add<cuFloatComplex>(const Tensor<cuFloatComplex> &tensorA,
                                      const Tensor<cuFloatComplex> &tensorB,
                                      Tensor<cuFloatComplex> &tensorC);

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype *a,
                           const Dtype *b, Dtype *c)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        c[index] = a[index] * b[index];
    }
}

template <typename Dtype>
__global__ void mul_with_stride_kernel(const int n, const Dtype *a,
                                       const Dtype *b, Dtype *c, const int stride)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        c[index] = a[index] * b[index / stride];
    }
}

template <typename Dtype>
void gpu_mul(const Tensor<Dtype> &tensorA,
             const Tensor<Dtype> &tensorB, Tensor<Dtype> &tensorC)
{
    if (tensorA.count() != tensorB.count())
    {
        if (tensorA.count() > tensorB.count())
        {
            CHECK_EQ(tensorA.count() % tensorB.count(), 0);
            if (tensorA.count() != tensorC.count() || tensorA.shape() != tensorC.shape())
            {
                LOG(FATAL) << "Trying to element-wise multiply Tensors to different size.";
            }
            int N = tensorA.count();
            mul_with_stride_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
                N, tensorA.gpu_data(), tensorB.gpu_data(), tensorC.mutable_gpu_data(),
                tensorA.count() / tensorB.count());
        }
        else
        {
            CHECK_EQ(tensorB.count() % tensorA.count(), 0);
            if (tensorB.count() != tensorC.count() || tensorB.shape() != tensorC.shape())
            {
                LOG(FATAL) << "Trying to element-wise multiply Tensors to different size.";
            }
            int N = tensorB.count();
            mul_with_stride_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
                N, tensorB.gpu_data(), tensorA.gpu_data(), tensorC.mutable_gpu_data(),
                tensorB.count() / tensorA.count());
        }
    }
    else
    {
        CHECK_EQ(tensorA.count(), tensorC.count());
        int N = tensorA.count();
        mul_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
            N, tensorA.gpu_data(), tensorB.gpu_data(), tensorC.mutable_gpu_data());
    }
}

// Instantiate the function
template void gpu_mul<float>(const Tensor<float> &tensorA,
                             const Tensor<float> &tensorB, Tensor<float> &tensorC);
template void gpu_mul<cuFloatComplex>(const Tensor<cuFloatComplex> &tensorA,
                                      const Tensor<cuFloatComplex> &tensorB,
                                      Tensor<cuFloatComplex> &tensorC);

template <typename Dtype>
__global__ void normalizing_kernel(const int n, Dtype *data, const int scale)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        data[index] /= scale;
    }
}

template <typename Dtype>
void gpu_normalizing(Tensor<Dtype> &tensor, const int &scale)
{
    int N = tensor.count();
    normalizing_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
        N, tensor.mutable_gpu_data(), scale);
}

// Instantiate the function
template void gpu_normalizing<float>(Tensor<float> &tensor, const int &scale);
template void gpu_normalizing<cuFloatComplex>(Tensor<cuFloatComplex> &tensor, const int &scale);

template <typename Dtype>
__global__ void reduce_kernel(const int n, const float *__restrict__ idata,
                              float *__restrict__ odata, const int stride)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        float tmp = 0;
        int init = index * stride;
        for (int i = init; i < init + stride; ++i)
        {
            tmp += idata[i];
        }
        odata[index] = tmp;
    }
}

// Cuz we can't overload assignment "=" operator - line 263
template <typename Dtype>
__global__ void reduce_kernel(const int n, const cuFloatComplex *__restrict__ in_data,
                              cuFloatComplex *__restrict__ out_data, const int stride)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        cuFloatComplex tmp = make_cuFloatComplex(0, 0);
        int init = index * stride;
        for (int i = init; i < init + stride; ++i)
        {
            tmp += in_data[i];
        }
        out_data[index] = tmp;
    }
}

template <typename Dtype>
void gpu_reduce(Tensor<Dtype> &input,
                Tensor<Dtype> &output, const int &axis)
{
    int num_axes = input.num_axes();
    // Move the axis to the end, making the data to reduce contiguous
    if (input.CanonicalAxisIndex(axis) != input.CanonicalAxisIndex(-1))
    {
        vector<int> transpose(num_axes);
        int i;
        for (i = 0; i < axis; ++i)
        {
            transpose[i] = i;
        }
        for (i = axis; i < num_axes - 1;)
        {
            transpose[i] = ++i;
        }
        transpose[i] = axis;
        input.transpose(transpose);
    }

    vector<int> shape = input.shape();
    shape[num_axes - 1] = 1;
    Tensor<Dtype> tmp(shape);
    int N = tmp.count();
    reduce_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
        N, input.gpu_data(), tmp.mutable_gpu_data(), input.shape(-1));

    // Recovery
    if (input.CanonicalAxisIndex(axis) != input.CanonicalAxisIndex(-1))
    {
        vector<int> transpose(num_axes);
        int i;
        for (i = 0; i < axis; ++i)
        {
            transpose[i] = i;
        }
        transpose[i] = num_axes - 1;
        for (++i; i < num_axes; ++i)
        {
            transpose[i] = i - 1;
        }
        input.transpose(transpose);
    }
    output.sharedata(tmp);
}

// Instantiate the function
template void gpu_reduce<float>(Tensor<float> &input,
                                Tensor<float> &output, const int &axis);
template void gpu_reduce<cuFloatComplex>(Tensor<cuFloatComplex> &input,
                                         Tensor<cuFloatComplex> &output, const int &axis);

} // namespace MRI
