#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cusparse.h>

#include "device_alternate.hpp"

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:                                   \
	classname(const classname &);          \
	classname &operator=(const classname &)

// Instantiate a class with float and cuFloatComplex specifications.
#define INSTANTIATE_CLASS(classname) \
	template class classname<float>; \
	template class classname<cuFloatComplex>

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

namespace MRI
{

// Common functions and classes from std that MRI often uses.
using std::fstream;
using std::ios;
using std::ostringstream;
using std::shared_ptr;
using std::unique_ptr;
using std::string;
using std::vector;

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int *pargc, char ***pargv);

// Find dir and if not, make it
void mk_dir(char *dir);

// A singleton class to hold common MRI stuff on GPU, such as the handler that
// GPU_Controller is going to use for cuFFT, cuSPARSE, etc.
class GPU_Controller
{
public:
	~GPU_Controller();

	static GPU_Controller &Get();

	// Getters for cuFFT and cuSPARSE handles
	inline static cufftHandle cuFFT_handle()
	{
		return Get().cuFFT_handle_;
	}
	inline static cusparseHandle_t cuSPARSE_handle()
	{
		return Get().cuSPARSE_handle_;
	}

	// The setters for the variables

	// Sets the device.
	static void SetDevice(const int device_id);

	// Prints the current GPU status.
	static void DeviceQuery();

	// Check if specified device is available
	static bool CheckDevice(const int device_id);

	// Search from start_id to the highest possible device ordinal,
	// Return the ordinal of the first available device.
	static int FindDevice(const int start_id = 0);

protected:
	cufftHandle cuFFT_handle_;
	cusparseHandle_t cuSPARSE_handle_;

private:
	// The private constructor to avoid duplicate instantiation.
	GPU_Controller();

	DISABLE_COPY_AND_ASSIGN(GPU_Controller);
};

} // namespace MRI

#endif // COMMON_HPP_
