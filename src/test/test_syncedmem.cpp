// g++ -std=c++11 test_syncedmem.cpp ../syncedmem.cpp ../common.cpp -I ../../include -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L ../../third_party -lglog -lgflags -lpthread -lcuda -lcusparse -lcufft -lcudart -o test_syncedmem

#include "syncedmem.hpp"

char log_path[] = "log";

enum SyncedHead
{
    UNINITIALIZED,
    HEAD_AT_CPU,
    HEAD_AT_GPU,
    SYNCED
};

using namespace std;
using namespace MRI;

int main(int argc, char **argv)
{
    GlobalInit(&argc, &argv);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_dir = log_path;
    mk_dir(log_path);

    float *ptr = (float *)malloc(100 * sizeof(float));
    memset(ptr, 1.0, 100 * sizeof(float));
    SyncedMemory *test = new SyncedMemory(100 * sizeof(float));
    CHECK_EQ(test->head(), UNINITIALIZED);
    test->set_cpu_data(ptr);
    CHECK_EQ(test->cpu_data(), ptr);
    CHECK_EQ(test->head(), HEAD_AT_CPU);

    ptr = (float *)test->gpu_data();
    CHECK_EQ(test->head(), SYNCED);
    float *ptr_ = (float *)test->mutable_gpu_data();
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, ptr_));
    CHECK_EQ(attributes.memoryType, cudaMemoryTypeDevice);
    CHECK_EQ(test->head(), HEAD_AT_GPU);

    float value = 2.0;
    CUDA_CHECK(cudaMemcpy(ptr_ + 10, &value, 1 * sizeof(float), cudaMemcpyHostToDevice));
    ptr = (float *)test->cpu_data();
    CHECK_EQ(ptr[10], 2.0);

    ptr = (float *)test->mutable_cpu_data();
    CHECK_EQ(test->head(), HEAD_AT_CPU);
    ptr[20] = 3.0;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    test->async_gpu_push(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CHECK_EQ(test->head(), SYNCED);
    ptr = (float *)test->gpu_data();
    CUDA_CHECK(cudaMemcpy(&value, ptr + 20, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_EQ(value, 3.0);
    delete test;
    printf("%s\n", "Testing passed.");

    return 0;
}