// nvcc -std=c++11 test_slice_concat.cpp ../slice_concat.cu ../tensor.cpp ../permute.cu ../syncedmem.cpp ../common.cpp -I ../../include -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L ../../third_party -lglog -lgflags -lpthread -lcuda -lcusparse -lcufft -lcudart -o test_slice_concat

#include "slice_concat.hpp"

char log_path[] = "log";

using namespace MRI;

// operator overload
inline bool operator==(const cuFloatComplex &A, const cuFloatComplex &B)
{
    return A.x == B.x && A.y == B.y;
}

// operator overload
inline bool operator==(const vector<int> &A, const vector<int> &B)
{
    if (A.size() != B.size())
        return false;
    for (int i = 0; i < A.size(); ++i)
    {
        if (A[i] != B[i])
            return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    GlobalInit(&argc, &argv);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_dir = log_path;
    mk_dir(log_path);

    // test gpu_slice
    vector<int> shape = {2, 3, 3, 3};
    Tensor<float> test1(shape);
    float *data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    test1.set_cpu_data(data);
    vector<shared_ptr<Tensor<float>>> v1;
    gpu_slice(test1, v1, -1);
    CHECK_EQ(v1.size(), 3);
    vector<int> new_shape = {2, 3, 3, 1};
    CHECK_EQ(v1[0]->shape() == new_shape, true);
    CHECK_EQ(v1[0]->data_at({1, 1, 1, 0}), 39.0);
    CHECK_EQ(v1[1]->data_at({1, 1, 1, 0}), 40.0);
    CHECK_EQ(v1[2]->data_at({1, 2, 2, 0}), 53.0);

    Tensor<cuFloatComplex> test2(shape);
    cuFloatComplex *data_ = (cuFloatComplex *)malloc(54 * sizeof(cuFloatComplex));
    for (int i = 0; i < 54; ++i)
        data_[i] = make_cuFloatComplex((float)i, 2 * (float)i);
    test2.set_cpu_data(data_);
    vector<shared_ptr<Tensor<cuFloatComplex>>> v2;
    gpu_slice(test2, v2, -1);
    CHECK_EQ(v2.size(), 3);
    new_shape = {2, 3, 3, 1};
    CHECK_EQ(v2[0]->shape() == new_shape, true);
    CHECK_EQ(v2[0]->data_at({1, 1, 1, 0}) == make_cuFloatComplex(39.0, 78.0), true);
    CHECK_EQ(v2[1]->data_at({1, 1, 1, 0}) == make_cuFloatComplex(40.0, 80.0), true);
    CHECK_EQ(v2[2]->data_at({1, 2, 2, 0}) == make_cuFloatComplex(53.0, 106.0), true);

    gpu_slice(test2, v2, 0);
    CHECK_EQ(v2.size(), 2);
    new_shape = {1, 3, 3, 3};
    CHECK_EQ(v2[0]->shape() == new_shape, true);
    CHECK_EQ(v2[0]->data_at({0, 1, 2, 2}) == make_cuFloatComplex(17.0, 34.0), true);
    CHECK_EQ(v2[0]->data_at({0, 2, 1, 1}) == make_cuFloatComplex(22.0, 44.0), true);
    CHECK_EQ(v2[1]->data_at({0, 2, 2, 0}) == make_cuFloatComplex(51.0, 102.0), true);
    CHECK_EQ(v2[1]->data_at({0, 1, 2, 2}) == make_cuFloatComplex(44.0, 88.0), true);

    test2.reshape({10, 10, 10});
    data_ = (cuFloatComplex *)malloc(1000 * sizeof(cuFloatComplex));
    for (int i = 0; i < 1000; ++i)
        data_[i] = make_cuFloatComplex((float)i, 2 * (float)i);
    test2.set_cpu_data(data_);
    vector<int> slice_points = {2, 4, 6, 9};
    gpu_slice(test2, v2, 0, slice_points);
    CHECK_EQ(v2.size(), 4);
    new_shape = {2, 10, 10};
    CHECK_EQ(v2[0]->shape() == new_shape, true);
    CHECK_EQ(v2[1]->shape() == new_shape, true);
    CHECK_EQ(v2[2]->shape() == new_shape, true);
    new_shape = {3, 10, 10};
    CHECK_EQ(v2[3]->shape() == new_shape, true);
    CHECK_EQ(v2[0]->data_at({0, 1, 2}) == make_cuFloatComplex(12.0, 24.0), true);
    CHECK_EQ(v2[0]->data_at({1, 5, 5}) == make_cuFloatComplex(155.0, 310.0), true);
    CHECK_EQ(v2[1]->data_at({0, 2, 2}) == make_cuFloatComplex(222.0, 444.0), true);
    CHECK_EQ(v2[1]->data_at({1, 3, 3}) == make_cuFloatComplex(333.0, 666.0), true);
    CHECK_EQ(v2[2]->data_at({1, 4, 2}) == make_cuFloatComplex(542.0, 1084.0), true);
    CHECK_EQ(v2[2]->data_at({1, 8, 9}) == make_cuFloatComplex(589.0, 1178.0), true);
    CHECK_EQ(v2[3]->data_at({2, 1, 1}) == make_cuFloatComplex(811.0, 1622.0), true);
    CHECK_EQ(v2[3]->data_at({2, 7, 7}) == make_cuFloatComplex(877.0, 1754.0), true);

    gpu_slice(test2, v2, 1, slice_points);
    CHECK_EQ(v2.size(), 4);
    new_shape = {10, 2, 10};
    CHECK_EQ(v2[0]->shape() == new_shape, true);
    CHECK_EQ(v2[1]->shape() == new_shape, true);
    CHECK_EQ(v2[2]->shape() == new_shape, true);
    new_shape = {10, 3, 10};
    CHECK_EQ(v2[3]->shape() == new_shape, true);
    CHECK_EQ(v2[0]->data_at({0, 1, 2}) == make_cuFloatComplex(12.0, 24.0), true);
    CHECK_EQ(v2[0]->data_at({5, 1, 5}) == make_cuFloatComplex(515.0, 1030.0), true);
    CHECK_EQ(v2[1]->data_at({3, 0, 5}) == make_cuFloatComplex(325.0, 650.0), true);
    CHECK_EQ(v2[1]->data_at({6, 1, 9}) == make_cuFloatComplex(639.0, 1278.0), true);
    CHECK_EQ(v2[2]->data_at({7, 0, 7}) == make_cuFloatComplex(747.0, 1494.0), true);
    CHECK_EQ(v2[2]->data_at({4, 0, 4}) == make_cuFloatComplex(444.0, 888.0), true);
    CHECK_EQ(v2[3]->data_at({2, 2, 2}) == make_cuFloatComplex(282.0, 564.0), true);
    CHECK_EQ(v2[3]->data_at({3, 2, 1}) == make_cuFloatComplex(381.0, 762.0), true);

    // test gpu_concat
    data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    vector<shared_ptr<Tensor<float>>> v;
    shared_ptr<Tensor<float>> ptr3(new Tensor<float>(shape));
    shared_ptr<Tensor<float>> ptr4(new Tensor<float>(shape));
    ptr3->set_cpu_data(data);
    ptr4->copyfrom(*ptr3);
    v.push_back(ptr3);
    v.push_back(ptr4);
    Tensor<float> test5;
    gpu_concat(v, test5, -1);
    CHECK_EQ(test5.data_at({1, 1, 1, 1}), 40.0);
    CHECK_EQ(test5.data_at({1, 1, 1, 4}), 40.0);
    CHECK_EQ(test5.data_at({1, 2, 2, 1}), 52.0);
    CHECK_EQ(test5.data_at({1, 2, 2, 5}), 53.0);

    data = (float *)malloc(54 * sizeof(float));
    for (int i = 0; i < 54; ++i)
        data[i] = (float)i;
    v.clear();
    shared_ptr<Tensor<float>> ptr6(new Tensor<float>(shape));
    shared_ptr<Tensor<float>> ptr7(new Tensor<float>(shape));
    ptr6->set_cpu_data(data);
    ptr7->copyfrom(*ptr6);
    v.push_back(ptr6);
    v.push_back(ptr7);
    Tensor<float> test8;
    gpu_concat(v, test8, -2);
    CHECK_EQ(test8.data_at({1, 1, 1, 1}), 40.0);
    CHECK_EQ(test8.data_at({1, 1, 4, 1}), 40.0);
    CHECK_EQ(test8.data_at({1, 2, 2, 1}), 52.0);
    CHECK_EQ(test8.data_at({1, 2, 5, 2}), 53.0);

    data_ = (cuFloatComplex *)malloc(54 * sizeof(cuFloatComplex));
    for (int i = 0; i < 54; ++i)
        data_[i] = make_cuFloatComplex((float)i, 2 * (float)i);
    vector<shared_ptr<Tensor<cuFloatComplex>>> v_;
    shared_ptr<Tensor<cuFloatComplex>> ptr9(new Tensor<cuFloatComplex>(shape));
    shared_ptr<Tensor<cuFloatComplex>> ptr10(new Tensor<cuFloatComplex>(shape));
    ptr9->set_cpu_data(data_);
    ptr10->copyfrom(*ptr9);
    v_.push_back(ptr9);
    v_.push_back(ptr10);
    Tensor<cuFloatComplex> test11;
    gpu_concat(v_, test11, -1);
    CHECK_EQ(test11.data_at({1, 1, 1, 1}) == make_cuFloatComplex(40.0, 80.0), true);
    CHECK_EQ(test11.data_at({1, 1, 1, 4}) == make_cuFloatComplex(40.0, 80.0), true);
    CHECK_EQ(test11.data_at({1, 2, 2, 1}) == make_cuFloatComplex(52.0, 104.0), true);
    CHECK_EQ(test11.data_at({1, 2, 2, 5}) == make_cuFloatComplex(53.0, 106.0), true);

    // test gpu_crop
    Tensor<float> test12({2, 3, 4, 5, 6});
    data = (float *)malloc(720 * sizeof(float));
    for (int i = 0; i < 720; ++i)
        data[i] = (float)i;
    test12.set_cpu_data(data);
    gpu_crop(test12, test12, {1, 2});
    new_shape = {1, 2, 4, 5, 6};
    CHECK_EQ(test12.shape() == new_shape, true);
    CHECK_EQ(test12.data_at({0, 1, 2, 2, 2}), 194.0);
    CHECK_EQ(test12.data_at({0, 0, 3, 4, 5}), 119.0);
    CHECK_EQ(test12.data_at({0, 1, 3, 1, 2}), 218.0);

    Tensor<float> test13({2, 3, 4, 5, 6});
    data = (float *)malloc(720 * sizeof(float));
    for (int i = 0; i < 720; ++i)
        data[i] = (float)i;
    test13.set_cpu_data(data);
    gpu_crop(test13, test13, {2, 2}, 2);
    new_shape = {2, 3, 2, 2, 6};
    CHECK_EQ(test13.shape() == new_shape, true);
    CHECK_EQ(test13.data_at({1, 1, 1, 1, 5}), 521.0);
    CHECK_EQ(test13.data_at({0, 1, 0, 0, 4}), 124.0);
    CHECK_EQ(test13.data_at({1, 0, 1, 1, 3}), 399.0);

    Tensor<cuFloatComplex> test14({2, 3, 4, 5, 6});
    data_ = (cuFloatComplex *)malloc(720 * sizeof(cuFloatComplex));
    for (int i = 0; i < 720; ++i)
        data_[i] = make_cuFloatComplex((float)i, 2 * (float)i);
    test14.set_cpu_data(data_);
    gpu_crop(test14, test14, {3, 3}, 3);
    new_shape = {2, 3, 4, 3, 3};
    CHECK_EQ(test14.shape() == new_shape, true);
    CHECK_EQ(test14.data_at({1, 2, 3, 1, 1}) == make_cuFloatComplex(697.0, 1394.0), true);
    CHECK_EQ(test14.data_at({1, 1, 2, 2, 2}) == make_cuFloatComplex(554.0, 1108.0), true);
    CHECK_EQ(test14.data_at({1, 0, 2, 2, 0}) == make_cuFloatComplex(432.0, 864.0), true);

    Tensor<cuFloatComplex> test15({10, 10, 10});
    data_ = (cuFloatComplex *)malloc(1000 * sizeof(cuFloatComplex));
    for (int i = 0; i < 1000; ++i)
        data_[i] = make_cuFloatComplex((float)i, 2 * (float)i);
    test15.set_cpu_data(data_);
    gpu_crop(test15, test15, {2, 2}, {3, 3}, 1);
    new_shape = {10, 3, 3};
    CHECK_EQ(test15.shape() == new_shape, true);
    CHECK_EQ(test15.data_at({3, 1, 1}) == make_cuFloatComplex(333.0, 666.0), true);
    CHECK_EQ(test15.data_at({3, 2, 1}) == make_cuFloatComplex(343.0, 686.0), true);
    CHECK_EQ(test15.data_at({5, 2, 0}) == make_cuFloatComplex(542.0, 1084.0), true);
    CHECK_EQ(test15.data_at({5, 2, 1}) == make_cuFloatComplex(543.0, 1086.0), true);
    CHECK_EQ(test15.data_at({7, 1, 0}) == make_cuFloatComplex(732.0, 1464.0), true);
    CHECK_EQ(test15.data_at({8, 2, 2}) == make_cuFloatComplex(844.0, 1688.0), true);

    printf("Testing passed.\n");

    return 0;
}