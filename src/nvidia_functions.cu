#include <cuda_runtime.h>
#include <cusparse.h>
#include <cufft.h>
#include <cufftXt.h>
#include <utility>

#include "nvidia_functions.hpp"

namespace MRI
{

template <typename Dtype>
__global__ void fftshift_kernel(const int n, const Dtype *__restrict__ in_data,
                                Dtype *__restrict__ out_data,
                                const int stride, const int offset)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        // relative index of output data in current interval
        int abs_idx_out = index % stride;
        // relative index of input data coresponding to output data in current interval
        int abs_idx_in = (abs_idx_out + offset) % stride;
        // absolute index of input data coresponding to output data
        int index_in = index + (abs_idx_in - abs_idx_out); //
        out_data[index] = in_data[index_in];
    }
}

// This is a naive implementation and only used as a backup
template <typename Dtype>
void gpu_fftshift(Tensor<Dtype> &tensor, const int &axis, const int &direction)
{
    if (tensor.shape(axis) == 1)
    {
        return;
    }

    int axis_ = tensor.CanonicalAxisIndex(axis);
    Tensor<Dtype> tmp;
    tmp.reshapelike(tensor);
    int stride = tensor.count(axis_);
    int offset = tensor.shape(axis_) / 2 * tensor.count(axis_ + 1);
    int N = tensor.count();
    if (direction == FORWARD)
    {
        fftshift_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
            N, tensor.gpu_data(), tmp.mutable_gpu_data(), stride, stride - offset);
    }
    else
    {
        fftshift_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
            N, tensor.gpu_data(), tmp.mutable_gpu_data(), stride, offset);
    }
    tensor.sharedata(tmp);
}

// Instantiate the function
template void gpu_fftshift<float>(Tensor<float> &tensor,
                                  const int &axis, const int &direction);
template void gpu_fftshift<cuFloatComplex>(Tensor<cuFloatComplex> &tensor,
                                           const int &axis, const int &direction);

// The callback API is available in the statically linked cuFFT library only,
// and only on 64 bit LINUX operating systems. Use of this API requires a current license.
/*
__device__ cuFloatComplex fast_fftshift_kernel(void *in_data, size_t offset,
                                               void *callerInfo, void *sharedptr)
{
    float a = (float)(1 - 2 * ((int)offset % 2));

    cuFloatComplex res = ((cuFloatComplex *)in_data)[offset];
    res.x = res.x * a;
    res.y = res.y * a;
    return res;
}

__device__ cufftCallbackLoadC fast_fftshift_ptr = fast_fftshift_kernel;

__device__ cuFloatComplex fast_ifftshift_kernel(void *in_data, size_t offset,
                                                void *callerInfo, void *sharedptr)
{
    float a = (float)(-1 + 2 * ((int)offset % 2));

    cuFloatComplex res = ((cuFloatComplex *)in_data)[offset];
    res.x = res.x * a;
    res.y = res.y * a;
    return res;
}

//__device__ cufftCallbackLoadC fast_ifftshift_ptr = fast_ifftshift_kernel;*/

__global__ void fast_fftshift_kernel(const int n,
                                     cuFloatComplex *in_data)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        float a = (float)(1 - 2 * ((int)index % 2));
        in_data[index].x *= a;
        in_data[index].y *= a;
    }
}

__global__ void fast_ifftshift_kernel(const int n,
                                      cuFloatComplex *in_data)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        float a = (float)(-1 + 2 * ((int)index % 2));
        in_data[index].x *= a;
        in_data[index].y *= a;
    }
}

template <>
void gpu_fft<cuFloatComplex>(Tensor<cuFloatComplex> &tensor, const vector<int> &axes,
                             const int &direction, const bool &shift)
{
    CHECK_GT(axes.size(), 0);
    cufftHandle plan;
    int stride = 1; // Stride of a single execution of FFT transform

    // 1D FFT transform
    if (axes.size() == 1)
    {
        CHECK_GE(tensor.num_axes(), 1);
        int axis = tensor.CanonicalAxisIndex(axes[0]);
        // Rearrange the data layout to make the input data contiguous
        if (axis != tensor.CanonicalAxisIndex(-1))
        {
            tensor.swapaxes(axis, -1);
        }

        stride = tensor.shape(-1);
        CUFFT_CHECK(cufftPlan1d(&plan, stride, CUFFT_C2C, tensor.count() / stride));
        if (shift)
        {
            // Fast fft shift kernel only works when stride is even
            // TODO: fix it
            if (stride % 2 == 0)
            {
                int N = tensor.count();
                if (direction == FORWARD)
                {
                    /*cufftCallbackLoadC host_fast_fftshift_ptr;
                    CUDA_CHECK(cudaMemcpyFromSymbol(&host_fast_fftshift_ptr, fast_fftshift_ptr,
                                                    sizeof(host_fast_fftshift_ptr)));
                    CUFFT_CHECK(cufftXtSetCallback(plan, (void **)&host_fast_fftshift_ptr,
                                                   CUFFT_CB_LD_COMPLEX, 0));*/
                    fast_fftshift_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
                        N, tensor.mutable_gpu_data());
                }
                else // Inverse
                {
                    /*cufftCallbackLoadC host_fast_ifftshift_ptr;
                    CUDA_CHECK(cudaMemcpyFromSymbol(&host_fast_ifftshift_ptr, fast_ifftshift_ptr,
                                                    sizeof(host_fast_ifftshift_ptr)));
                    CUFFT_CHECK(cufftXtSetCallback(plan, (void **)&host_fast_ifftshift_ptr,
                                                   CUFFT_CB_LD_COMPLEX, 0));*/
                    fast_ifftshift_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
                        N, tensor.mutable_gpu_data());
                }
            }
            else
            {
                gpu_fftshift(tensor, -1, direction);
            }
        }
        CUFFT_CHECK(cufftExecC2C(plan, tensor.mutable_gpu_data(),
                                 tensor.mutable_gpu_data(), direction));
        if (shift)
        {
            if (stride % 2 == 0)
            {
                int N = tensor.count();
                if (direction == FORWARD)
                {
                    fast_ifftshift_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
                        N, tensor.mutable_gpu_data());
                }
                else // Inverse
                {
                    fast_fftshift_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
                        N, tensor.mutable_gpu_data());
                }
            }
            else
            {
                gpu_fftshift(tensor, -1, direction);
            }
        }

        // Recovery
        if (axis != tensor.CanonicalAxisIndex(-1))
        {
            tensor.swapaxes(axis, -1);
        }
    }
    // 2D FFT transform
    else
    {
        CHECK_GE(tensor.num_axes(), 1);
        int axis1 = tensor.CanonicalAxisIndex(axes[0]);
        int axis2 = tensor.CanonicalAxisIndex(axes[1]);
        CHECK_NE(axis1, axis2);
        if (axis1 > axis2)
        {
            std::swap(axis1, axis2);
        }
        if (axis1 != tensor.CanonicalAxisIndex(-2) || axis2 != tensor.CanonicalAxisIndex(-1))
        {
            vector<int> shape;
            for (int i = 0; i < tensor.num_axes(); ++i)
            {
                if (i != axis1 && i != axis2)
                {
                    shape.push_back(i);
                }
            }
            shape.push_back(axis1);
            shape.push_back(axis2);
            tensor.transpose(shape);
        }

        stride = tensor.shape(-2) * tensor.shape(-1);
        int n[2] = {tensor.shape(-2), tensor.shape(-1)};
        CUFFT_CHECK(cufftPlanMany(&plan, 2, n,
                                  NULL, 1, 0,
                                  NULL, 1, 0,
                                  CUFFT_C2C, tensor.count() / stride));

        if (shift)
        {
            // TODO: 2D FFT shift
            NOT_IMPLEMENTED;
        }
        CUFFT_CHECK(cufftExecC2C(plan, tensor.mutable_gpu_data(),
                                 tensor.mutable_gpu_data(), direction));

        // Recovery
        if (axis1 != tensor.CanonicalAxisIndex(-2) || axis2 != tensor.CanonicalAxisIndex(-1))
        {
            int num_axes = tensor.num_axes();
            vector<int> shape(num_axes, -1);
            shape[axis1] = num_axes - 2;
            shape[axis2] = num_axes - 1;
            for (int i = 0, j = 0; i < num_axes; ++i)
            {
                if (shape[i] == -1)
                {
                    shape[i] = j;
                    ++j;
                }
            }
            tensor.transpose(shape);
        }
    }
    gpu_normalizing(tensor, stride);
    CUFFT_CHECK(cufftDestroy(plan));
}

template <>
void gpu_dot<cuFloatComplex>(const SparseTensor<cuFloatComplex> &tensorA,
                             Tensor<cuFloatComplex> &tensorB, Tensor<cuFloatComplex> &tensorC)
{
    // tensorB.shape() can be {row, col, 1} due to the slice operation
    CHECK_EQ(tensorA.num_axes(), 2);
    CHECK_GE(tensorB.num_axes(), 2);
    for (int i = 2; i < tensorB.num_axes(); ++i)
    {
        CHECK_EQ(tensorB.shape(i), 1);
    }
    // Match the data layout format of cuSPARSE
    tensorB.swapaxes(0, 1);
    vector<int> shape = {tensorB.shape(0), tensorA.row()};
    tensorC.reshape(shape);

    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;
    cuFloatComplex alpha = make_cuFloatComplex(1.0, 0.0);
    cuFloatComplex beta = make_cuFloatComplex(0.0, 0.0);
    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseCcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  tensorA.row(), tensorB.shape(0), tensorA.col(),
                                  tensorA.nnz(), &alpha, descr, tensorA.gpu_data(),
                                  tensorA.gpu_ptr(), tensorA.gpu_indices(),
                                  tensorB.gpu_data(), tensorB.shape(1),
                                  &beta, tensorC.mutable_gpu_data(), tensorA.row()));

    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    tensorC.swapaxes(0, 1);

    // Recovery - not needed in this case
    // tensorB.swapaxes(0,1);
}

} // namespace MRI
