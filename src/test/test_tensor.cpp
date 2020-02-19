// nvcc -std=c++11 test_tensor.cpp ../tensor.cpp ../permute.cu ../syncedmem.cpp ../common.cpp -I ../../include -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L ../../third_party -lglog -lgflags -lpthread -lcuda -lcusparse -lcufft -lcudart -o test_tensor

#include "tensor.hpp"

char log_path[] = "log";

using namespace MRI;
using namespace std;

int main(int argc, char **argv)
{
    GlobalInit(&argc, &argv);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_dir = log_path;
    mk_dir(log_path);

    Tensor<float> test1;
    LOG(INFO) << test1.shape_string();
    test1.~Tensor();

    vector<int> shape = {1, 2, 3, 4, 5};
    float *data = (float *)malloc(120 * sizeof(float));
    for (int i = 0; i < 120; ++i)
        data[i] = (float)i;
    Tensor<float> test2(shape);
    test2.set_cpu_data(data);
    LOG(INFO) << test2.shape_string();
    CHECK_EQ(test2.shape().size(), shape.size());
    for (int i = 0; i < test2.shape().size(); ++i)
        CHECK_EQ(test2.shape(i), shape[i]);
    CHECK_EQ(test2.shape(3), 4);
    CHECK_EQ(test2.num_axes(), 5);
    CHECK_EQ(test2.count(), 120);
    CHECK_EQ(test2.count(2, 5), 60);
    CHECK_EQ(test2.CanonicalAxisIndex(-1), 4);
    vector<int> index = {0, 1, 2, 3, 0};
    CHECK_EQ(test2.data_at(index), 115.0);
    const float *ptr = test2.gpu_data();
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, ptr));
    CHECK_EQ(attributes.memoryType, cudaMemoryTypeDevice);
    vector<int> new_shape = {3, 4, 10};
    test2.reshape(new_shape);
    LOG(INFO) << test2.shape_string();
    CHECK_EQ(test2.shape().size(), new_shape.size());
    for (int i = 0; i < test2.shape().size(); ++i)
        CHECK_EQ(test2.shape(i), new_shape[i]);

    Tensor<float> test3(shape);
    test3.copyfrom(test2);
    test3.reshapelike(test2);
    CHECK_EQ(test3.shape().size(), new_shape.size());
    for (int i = 0; i < test3.shape().size(); ++i)
        CHECK_EQ(test3.shape(i), new_shape[i]);
    vector<int> index_ = {1, 2, 5};
    CHECK_EQ(test3.data_at(index_), 65.0);
    test2.reshape(shape);
    test3.reshapelike(test2);
    CHECK_EQ(test3.shape().size(), shape.size());
    for (int i = 0; i < test3.shape().size(); ++i)
        CHECK_EQ(test3.shape(i), shape[i]);
    for (int i = 0; i < test3.shape().size(); ++i)
        CHECK_EQ(test3.shape(i), shape[i]);

    Tensor<float> test4(shape);
    test4.sharedata(test3);
    CHECK_EQ(test4.data_at(index), 115.0);
    test4.~Tensor();

    Tensor<cuFloatComplex> *test5 = new Tensor<cuFloatComplex>(shape);
    delete test5;

    printf("Testing passed.\n");

    return 0;
}