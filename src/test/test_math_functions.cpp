// nvcc -std=c++11 test_math_functions.cpp ../math_functions.cu ../tensor.cpp ../permute.cu ../syncedmem.cpp ../common.cpp -I ../../include -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L ../../third_party -lglog -lgflags -lpthread -lcuda -lcusparse -lcufft -lcudart -o test_math_functions

#include "math_functions.hpp"

char log_path[] = "log";

using namespace MRI;

// operator overload
inline bool operator==(const cuFloatComplex &A, const cuFloatComplex &B)
{
    return A.x == B.x && A.y == B.y;
}

int main(int argc, char **argv)
{
    GlobalInit(&argc, &argv);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_dir = log_path;
    mk_dir(log_path);

    vector<int> shape = {2, 3, 3, 3};
    Tensor<float> test1(shape);
    gpu_set(test1, (float)2.0);
    CHECK_EQ(test1.data_at({1, 1, 1, 1}), 2.0);
    CHECK_EQ(test1.data_at({1, 2, 2, 2}), 2.0);

    Tensor<cuFloatComplex> test2(shape);
    cuFloatComplex value = make_cuFloatComplex(1.0, 2.0);
    gpu_set(test2, value);
    CHECK_EQ(test2.data_at({1, 1, 1, 1}) == value, true);
    CHECK_EQ(test2.data_at({1, 2, 2, 2}) == value, true);

    Tensor<float> test3(shape);
    float *data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test3.set_cpu_data(data);
    gpu_add(test1, test3, test3);
    CHECK_EQ(test3.data_at({1, 1, 1, 1}), 42.0);
    CHECK_EQ(test3.data_at({1, 2, 2, 2}), 55.0);

    Tensor<cuFloatComplex> test4(shape);
    cuFloatComplex *data_ = (cuFloatComplex *)malloc(54 * sizeof(cuFloatComplex));
    for (int i = 0; i < 54; ++i)
        data_[i] = make_cuFloatComplex((float)i, (float)i);
    test4.set_cpu_data(data_);
    gpu_add(test2, test4, test4);
    CHECK_EQ(test4.data_at({1, 1, 1, 1}) == make_cuFloatComplex(41.0, 42.0), true);
    CHECK_EQ(test4.data_at({1, 2, 2, 2}) == make_cuFloatComplex(54.0, 55.0), true);

    Tensor<float> test5(shape);
    data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test5.set_cpu_data(data);
    gpu_mul(test1, test5, test5);
    CHECK_EQ(test5.data_at({1, 1, 1, 1}), 80.0);
    CHECK_EQ(test5.data_at({1, 2, 2, 2}), 106.0);

    Tensor<cuFloatComplex> test6(shape);
    data_ = (cuFloatComplex *)malloc(54 * sizeof(cuFloatComplex));
    for (int i = 0; i < 54; ++i)
        data_[i] = make_cuFloatComplex((float)i, (float)i);
    test6.set_cpu_data(data_);
    gpu_mul(test2, test6, test6);
    CHECK_EQ(test6.data_at({1, 1, 1, 1}) == make_cuFloatComplex(-40.0, 120.0), true);
    CHECK_EQ(test6.data_at({1, 2, 2, 2}) == make_cuFloatComplex(-53.0, 159.0), true);

    Tensor<float> test7(shape);
    data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test7.set_cpu_data(data);
    gpu_normalizing(test7, 2);
    CHECK_EQ(test7.data_at({1, 1, 1, 1}), 20.0);
    CHECK_EQ(test7.data_at({1, 2, 2, 2}), 26.5);

    Tensor<cuFloatComplex> test8(shape);
    data_ = (cuFloatComplex *)malloc(54 * sizeof(cuFloatComplex));
    for (int i = 0; i < 54; ++i)
        data_[i] = make_cuFloatComplex((float)i, (float)i);
    test8.set_cpu_data(data_);
    gpu_normalizing(test8, 2);
    CHECK_EQ(test8.data_at({1, 1, 1, 1}) == make_cuFloatComplex(20.0, 20.0), true);
    CHECK_EQ(test8.data_at({1, 2, 2, 2}) == make_cuFloatComplex(26.5, 26.5), true);

    Tensor<float> test9(shape);
    data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test9.set_cpu_data(data);
    Tensor<float> test10;
    gpu_reduce(test9, test10, -1);
    CHECK_EQ(test10.data_at({1, 2, 2}), 156.0);
    CHECK_EQ(test10.data_at({0, 2, 1}), 66.0);

    Tensor<cuFloatComplex> test11(shape);
    data_ = (cuFloatComplex *)malloc(54 * sizeof(cuFloatComplex));
    for (int i = 0; i < 54; ++i)
        data_[i] = make_cuFloatComplex((float)i, 2 * (float)i);
    test11.set_cpu_data(data_);
    Tensor<cuFloatComplex> test12;
    gpu_reduce(test11, test12, -1);
    CHECK_EQ(test12.shape(3), 1);
    CHECK_EQ(test12.data_at({1, 2, 2, 0}) == make_cuFloatComplex(156.0, 312.0), true);
    CHECK_EQ(test12.data_at({0, 2, 1, 0}) == make_cuFloatComplex(66.0, 132.0), true);

    Tensor<cuFloatComplex> test13;
    gpu_reduce(test2, test13, -1);
    gpu_mul(test13, test6, test6);
    CHECK_EQ(test6.data_at({1, 1, 1, 1}) == make_cuFloatComplex(-840.0, 120.0), true);
    CHECK_EQ(test6.data_at({1, 2, 2, 2}) == make_cuFloatComplex(-1113.0, 159.0), true);

    printf("Testing passed.\n");

    return 0;
}