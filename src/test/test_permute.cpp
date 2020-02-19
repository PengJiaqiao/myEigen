// nvcc -std=c++11 test_permute.cpp ../tensor.cpp ../permute.cu ../syncedmem.cpp ../common.cpp -I ../../include -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L ../../third_party -lglog -lgflags -lpthread -lcuda -lcusparse -lcufft -lcudart -o test_permute

#include "permute.hpp"

char log_path[] = "log";

using namespace MRI;

int main(int argc, char **argv)
{
    GlobalInit(&argc, &argv);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_dir = log_path;
    mk_dir(log_path);

    vector<int> shape = {2, 3, 3, 3};
    Tensor<float> test1(shape);
    float *data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test1.set_cpu_data(data);
    transpose(test1, {3, 2, 1, 0});
    CHECK_EQ(test1.data_at({2, 2, 1, 0}), 17);
    CHECK_EQ(test1.data_at({1, 2, 2, 1}), 52);
    CHECK_EQ(test1.data_at({0, 1, 2, 1}), 48);

    test1.reshape(shape);
    data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test1.set_cpu_data(data);
    swapaxes(test1, 1, 2);
    CHECK_EQ(test1.data_at({1, 1, 2, 1}), 49);
    CHECK_EQ(test1.data_at({0, 1, 2, 2}), 23);
    CHECK_EQ(test1.data_at({1, 2, 0, 0}), 33);

    test1.reshape(shape);
    data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test1.set_cpu_data(data);
    test1.transpose({3, 1, 0, 2});
    CHECK_EQ(test1.data_at({2, 2, 1, 0}), 47);
    CHECK_EQ(test1.data_at({1, 2, 1, 1}), 49);
    CHECK_EQ(test1.data_at({0, 1, 0, 1}), 12);

    test1.reshape(shape);
    data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test1.set_cpu_data(data);
    test1.swapaxes(0, 2);
    CHECK_EQ(test1.data_at({1, 1, 0, 1}), 13);
    CHECK_EQ(test1.data_at({0, 1, 0, 2}), 11);
    CHECK_EQ(test1.data_at({1, 2, 0, 0}), 21);

    test1.reshape(shape);
    data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test1.set_cpu_data(data);
    test1.swapaxes(0, -2);
    CHECK_EQ(test1.data_at({1, 1, 0, 1}), 13);
    CHECK_EQ(test1.data_at({0, 1, 0, 2}), 11);
    CHECK_EQ(test1.data_at({1, 2, 0, 0}), 21);

    printf("Testing passed.\n");

    return 0;
}