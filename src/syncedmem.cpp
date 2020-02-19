#include "common.hpp"
#include "syncedmem.hpp"

namespace MRI
{
SyncedMemory::SyncedMemory()
	: cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(0), head_(UNINITIALIZED),
	  cpu_malloc_use_cuda_(true), own_cpu_data_(false), own_gpu_data_(false),
	  device_(0) {}

SyncedMemory::SyncedMemory(size_t size)
	: cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(size), head_(UNINITIALIZED),
	  cpu_malloc_use_cuda_(true), own_cpu_data_(false), own_gpu_data_(false),
	  device_(0) {}

SyncedMemory::~SyncedMemory()
{
	check_device();
	if (cpu_ptr_ && own_cpu_data_)
	{
		FreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
	}
	if (gpu_ptr_ && own_gpu_data_)
	{
		CUDA_CHECK(cudaFree(gpu_ptr_));
	}
}

// Synchronize data from GPU to CPU
inline void SyncedMemory::to_cpu()
{
	switch (head_)
	{
	case UNINITIALIZED:
		MallocHost(&cpu_ptr_, size_, cpu_malloc_use_cuda_);
		memset(cpu_ptr_, 0, size_);
		head_ = HEAD_AT_CPU;
		own_cpu_data_ = true;
		break;
	case HEAD_AT_GPU:
		if (cpu_ptr_ == nullptr)
		{
			MallocHost(&cpu_ptr_, size_, cpu_malloc_use_cuda_);
			own_cpu_data_ = true;
		}
		CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDefault));
		head_ = SYNCED;
		break;
	case HEAD_AT_CPU:
		break;
	case SYNCED:
		break;
	}
	return;
}

// Synchronize data from CPU to GPU
inline void SyncedMemory::to_gpu()
{
	switch (head_)
	{
	case UNINITIALIZED:
		CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
		CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
		head_ = HEAD_AT_GPU;
		own_gpu_data_ = true;
		break;
	case HEAD_AT_CPU:
		if (gpu_ptr_ == nullptr)
		{
			CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
			own_gpu_data_ = true;
		}
		CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyDefault));
		head_ = SYNCED;
		break;
	case HEAD_AT_GPU:
		break;
	case SYNCED:
		break;
	}
	return;
}

// Getter of the address of the CPU data
const void *SyncedMemory::cpu_data()
{
	to_cpu();
	return (const void *)cpu_ptr_;
}

// The pointer of the original CPU data now points to where the data points to
// Release the memory of the original CPU data if own CPU data
void SyncedMemory::set_cpu_data(void *data)
{
	CHECK(data);
	if (own_cpu_data_)
	{
		FreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
	}
	cpu_ptr_ = data;
	head_ = HEAD_AT_CPU;
	own_cpu_data_ = false;
	return;
}

// Getter of the address of the GPU data
const void *SyncedMemory::gpu_data()
{
	to_gpu();
	return (const void *)gpu_ptr_;
}

// The pointer of the original GPU data now points to where the data points to
// Release the memory of the original GPU data if own GPU data
void SyncedMemory::set_gpu_data(void *data)
{
	CHECK(data);
	if (own_gpu_data_)
	{
		CUDA_CHECK(cudaFree(gpu_ptr_));
	}
	gpu_ptr_ = data;
	head_ = HEAD_AT_GPU;
	own_gpu_data_ = false;
	return;
}

// Getter of the address of the CPU data - for writing data
void *SyncedMemory::mutable_cpu_data()
{
	to_cpu();
	head_ = HEAD_AT_CPU; // Need synchronization
	return cpu_ptr_;
}

// The same as above
void *SyncedMemory::mutable_gpu_data()
{
	to_gpu();
	head_ = HEAD_AT_GPU;
	return gpu_ptr_;
}

// Asynchronous transfer of CUDA Memory copy
void SyncedMemory::async_gpu_push(const cudaStream_t &stream)
{
	CHECK(head_ == HEAD_AT_CPU);
	if (gpu_ptr_ == nullptr)
	{
		CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
		own_gpu_data_ = true;
	}
	const cudaMemcpyKind put = cudaMemcpyHostToDevice;
	CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
	// Assume caller will synchronize on the stream before use
	head_ = SYNCED;
}

// Check current device and whether gpu_ptr_ points to the device obtained during class construction
void SyncedMemory::check_device()
{
	int device;
	cudaGetDevice(&device);
	CHECK(device == device_);
	if (gpu_ptr_ && own_gpu_data_)
	{
		cudaPointerAttributes attributes;
		CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
		CHECK(attributes.device == device_);
	}
	return;
}

} // namespace MRI
