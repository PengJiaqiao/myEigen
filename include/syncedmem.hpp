#ifndef SYNCEDMEM_HPP_
#define SYNCEDMEM_HPP_

#include "common.hpp"

namespace MRI
{

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).

inline void MallocHost(void **ptr, size_t size, const bool &use_cuda)
{
	if (use_cuda == true)
	{
		CUDA_CHECK(cudaMallocHost(ptr, size));
		return;
	}
	// use malloc if without CUDA
	*ptr = malloc(size);
	CHECK(*ptr) << "host allocation of size " << size << " failed";
	return;
}

inline void FreeHost(void *ptr, const bool &use_cuda)
{
	if (use_cuda)
	{
		CUDA_CHECK(cudaFreeHost(ptr));
		return;
	}
	free(ptr);
	return;
}

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *		  Inspired by Caffe. https://github.com/BVLC/caffe
 */

class SyncedMemory
{
public:
	SyncedMemory();
	explicit SyncedMemory(size_t size);
	~SyncedMemory();

	const void *cpu_data();		   // Getter of the address of the CPU data
	void set_cpu_data(void *data); // set CPU data by pointer 'data'
	const void *gpu_data();		   // Getter of the address of the GPU data
	void set_gpu_data(void *data); // set GPU data by pointer 'data'

	// For writing data
	void *mutable_cpu_data();
	void *mutable_gpu_data();

	// Four different state of the data
	enum SyncedHead
	{
		UNINITIALIZED,
		HEAD_AT_CPU,
		HEAD_AT_GPU,
		SYNCED
	};
	SyncedHead head() const { return head_; }
	size_t size() const { return size_; }

	// Asynchronous transfer of CUDA Memory copy
	void async_gpu_push(const cudaStream_t &stream);

private:
	void check_device();
	void to_cpu(); // Synchronize data from GPU to CPU
	void to_gpu(); // Synchronize data from CPU to GPU

	void *cpu_ptr_; // CPU data pointer
	void *gpu_ptr_; // GPU data pointer

	size_t size_;
	SyncedHead head_; // State machine variable

	bool cpu_malloc_use_cuda_;
	bool own_cpu_data_;
	bool own_gpu_data_;

	int device_; // GPU device ID

	DISABLE_COPY_AND_ASSIGN(SyncedMemory);
}; // class SyncedMemory

} // namespace MRI

#endif // SYNCEDMEM_HPP_
