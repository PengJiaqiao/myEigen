#include <glog/logging.h>
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "common.hpp"

#define MODE (S_IRWXU | S_IRWXG | S_IRWXO)

namespace MRI
{

void GlobalInit(int *pargc, char ***pargv)
{
	// Google flags.
	::gflags::ParseCommandLineFlags(pargc, pargv, true);
	// Google logging.
	::google::InitGoogleLogging(*(pargv)[0]);
	// Provide a backtrace on segfault.
	::google::InstallFailureSignalHandler();
}

void mk_dir(char *dir)
{
	DIR *mydir = nullptr;
	if ((mydir = opendir(dir)) == nullptr)
	{
		int ret = mkdir(dir, MODE);
		if (ret != 0)
		{
			LOG(ERROR) << dir << "created failed.";
		}
		LOG(INFO) << dir << "created success.";
	}
}

static unique_ptr<GPU_Controller> GPU_Controller_instance_;

GPU_Controller &GPU_Controller::Get()
{
	if (!GPU_Controller_instance_.get())
	{
		GPU_Controller_instance_.reset(new GPU_Controller());
	}
	return *(GPU_Controller_instance_.get());
}

GPU_Controller::GPU_Controller()
	: cuFFT_handle_(0), cuSPARSE_handle_(nullptr)
{
	// Try to create a cuFFT handler, and report an error if failed.
	if (cufftCreate(&cuFFT_handle_) != CUFFT_SUCCESS)
	{
		LOG(ERROR) << "Cannot create cuFFT handle. cuFFT won't be available.";
		exit(0);
	}
	// Try to create a cuSPARSE handler, and report an error if failed.
	if (cusparseCreate(&cuSPARSE_handle_) != CUSPARSE_STATUS_SUCCESS)
	{
		LOG(ERROR) << "Cannot create cuSPARSE handle. cuSPARSE won't be available.";
		exit(0);
	}
}

GPU_Controller::~GPU_Controller()
{
	if (cuFFT_handle_)
		CUFFT_CHECK(cufftDestroy(cuFFT_handle_));
	if (cuSPARSE_handle_)
		CUSPARSE_CHECK(cusparseDestroy(cuSPARSE_handle_));
}

void GPU_Controller::SetDevice(const int device_id)
{
	int current_device;
	CUDA_CHECK(cudaGetDevice(&current_device));
	if (current_device == device_id)
	{
		return;
	}
	// The call to cudaSetDevice must come before any calls to Get, which
	// may perform initialization using the GPU.
	CUDA_CHECK(cudaSetDevice(device_id));
}

void GPU_Controller::DeviceQuery()
{
	cudaDeviceProp prop;
	int device;
	if (cudaSuccess != cudaGetDevice(&device))
	{
		printf("No cuda device present.\n");
		return;
	}
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
	LOG(INFO) << "Device id:                     " << device;
	LOG(INFO) << "Major revision number:         " << prop.major;
	LOG(INFO) << "Minor revision number:         " << prop.minor;
	LOG(INFO) << "Name:                          " << prop.name;
	LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
	LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
	LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
	LOG(INFO) << "Warp size:                     " << prop.warpSize;
	LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
	LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
	LOG(INFO) << "Maximum dimension of block:    "
			  << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
			  << prop.maxThreadsDim[2];
	LOG(INFO) << "Maximum dimension of grid:     "
			  << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
			  << prop.maxGridSize[2];
	LOG(INFO) << "Clock rate:                    " << prop.clockRate;
	LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
	LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
	LOG(INFO) << "Concurrent copy and execution: "
			  << (prop.deviceOverlap ? "Yes" : "No");
	LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
	LOG(INFO) << "Kernel execution timeout:      "
			  << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
	return;
}

bool GPU_Controller::CheckDevice(const int device_id)
{
	// This function checks the availability of GPU #device_id.
	// It attempts to create a context on the device by calling cudaFree(0).
	// cudaSetDevice() alone is not sufficient to check the availability.
	// It lazily records device_id, however, does not initialize a
	// context. So it does not know if the host thread has the permission to use
	// the device or not.
	//
	// In a shared environment where the devices are set to EXCLUSIVE_PROCESS
	// or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
	// even if the device is exclusively occupied by another process or thread.
	// Cuda operations that initialize the context are needed to check
	// the permission. cudaFree(0) is one of those with no side effect,
	// except the context initialization.
	bool r = ((cudaSuccess == cudaSetDevice(device_id)) &&
			  (cudaSuccess == cudaFree(0)));
	// reset any error that may have occurred.
	cudaGetLastError();
	return r;
}

int GPU_Controller::FindDevice(const int start_id)
{
	// This function finds the first available device by checking devices with
	// ordinal from start_id to the highest available value. In the
	// EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
	// claims the device due to the initialization of the context.
	int count = 0;
	CUDA_CHECK(cudaGetDeviceCount(&count));
	for (int i = start_id; i < count; i++)
	{
		if (CheckDevice(i))
			return i;
	}
	return -1;
}

const char *cufftGetErrorString(cufftResult error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "cuFFT successfully created the FFT plan";
	case CUFFT_ALLOC_FAILED:
		return "The allocation of resources for the plan failed";
	case CUFFT_INVALID_VALUE:
		return "One or more invalid parameters were passed to the API";
	case CUFFT_INTERNAL_ERROR:
		return "An internal driver error was detected";
	case CUFFT_SETUP_FAILED:
		return "The cuFFT library failed to initialize";
	case CUFFT_NOT_IMPLEMENTED:
		return "The callback API is available in the statically linked cuFFT library only, and only on 64 bit LINUX operating systems. Use of this API requires a current license.";
	case CUFFT_LICENSE_ERROR:
		return "A valid license file required.";
	}
	return "Unknown cufft status";
}

const char *cusparseGetErrorString(cusparseStatus_t error)
{
	switch (error)
	{
	case CUSPARSE_STATUS_SUCCESS:
		return "The operation completed successfully";
	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "The cuSPARSE library was not initialized";
	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "Resource allocation failed inside the cuSPARSE library";
	case CUSPARSE_STATUS_INVALID_VALUE:
		return "An unsupported value or parameter was passed to the function";
	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "The function requires a feature absent from the device architecture";
	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "An access to GPU memory space failed";
	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "The GPU program failed to execute";
	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "An internal cuSPARSE operation failed";
	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "The matrix type is not supported by this function";
	case CUSPARSE_STATUS_ZERO_PIVOT:
		return "An entry of the matrix is either structural zero or numerical zero (singular block)";
	}
	return "Unknown cuSPARSE status";
}

} // namespace MRI
