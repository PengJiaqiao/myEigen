#ifndef DEVICE_ALTERNATE_HPP_
#define DEVICE_ALTERNATE_HPP_

#include <driver_types.h> // cuda driver types
#include <glog/logging.h>

/**
 * @brief Maroc for GPU device 
 * 		  Various checks for different function calls
 */

#define CUDA_CHECK(func)                                                  \
	do                                                                    \
	{                                                                     \
		cudaError_t error = (func);                                       \
		CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
	} while (0)

#define CUFFT_CHECK(func)                                                    \
	do                                                                       \
	{                                                                        \
		cufftResult status = (func);                                         \
		CHECK_EQ(status, CUFFT_SUCCESS) << " "                               \
										<< MRI::cufftGetErrorString(status); \
	} while (0)

#define CUSPARSE_CHECK(func)                                                              \
	do                                                                                    \
	{                                                                                     \
		cusparseStatus_t status = (func);                                                 \
		CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS) << " "                                  \
												  << MRI::cusparseGetErrorString(status); \
	} while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		 i < (n);                                       \
		 i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace MRI
{

// CUDA: library error reporting.
const char *cufftGetErrorString(cufftResult error);
const char *cusparseGetErrorString(cusparseStatus_t error);

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N)
{
	return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// Call cudaMemcpy
inline void MRI_gpu_memcpy(void *dst, const void *src, const size_t count)
{
	if (dst != src)
	{
		CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDefault));
	}
}

} // namespace MRI

#endif // DEVICE_ALTERNATE_HPP_