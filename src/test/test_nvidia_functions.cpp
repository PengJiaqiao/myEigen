// nvcc -std=c++11 test_nvidia_functions.cpp ../nvidia_functions.cu ../math_functions.cu ../sparse_tensor.cpp ../tensor.cpp ../permute.cu ../syncedmem.cpp ../common.cpp -I ../../include -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L ../../third_party -lglog -lgflags -lpthread -lcuda -lcusparse -lcufft -lcudart -o test_nvidia_functions

#include "nvidia_functions.hpp"

char log_path[] = "log";

using namespace MRI;

// operator overload
inline bool operator==(const cuFloatComplex &A, const cuFloatComplex &B)
{
    float distance = pow(A.x - B.x, 2) + pow(A.y - B.y, 2);
    float length = pow(B.x, 2) + pow(B.y, 2);
    bool res;
    if (length >= 0.000001)
        res = (distance / length) <= 0.001;
    else
        res = distance <= 0.000001;
    return res;
}

int main(int argc, char **argv)
{
    GlobalInit(&argc, &argv);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_dir = log_path;
    mk_dir(log_path);

    // test gpu_fftshift
    vector<int> shape = {3, 3, 3};
    Tensor<float> test1(shape);
    float *data = (float *)malloc(27 * sizeof(float));
    for (int i = 0; i < 27; ++i)
        data[i] = (float)i;
    test1.set_cpu_data(data);
    Tensor<float> test2(shape);
    test2.copyfrom(test1);
    gpu_fftshift(test2, 0, INVERSE);
    CHECK_EQ(test2.data_at({0, 1, 2}), 14.0);
    CHECK_EQ(test2.data_at({1, 1, 2}), 23.0);
    CHECK_EQ(test2.data_at({2, 1, 2}), 5.0);

    Tensor<float> test3(shape);
    test3.copyfrom(test1);
    gpu_fftshift(test3, 1, INVERSE);
    CHECK_EQ(test3.data_at({0, 1, 2}), 8.0);
    CHECK_EQ(test3.data_at({1, 1, 2}), 17.0);
    CHECK_EQ(test3.data_at({2, 1, 2}), 26.0);

    Tensor<float> test4(shape);
    test4.copyfrom(test1);
    gpu_fftshift(test4, 2, INVERSE);
    CHECK_EQ(test4.data_at({0, 1, 2}), 3.0);
    CHECK_EQ(test4.data_at({1, 1, 2}), 12.0);
    CHECK_EQ(test4.data_at({2, 1, 2}), 21.0);

    Tensor<float> test5(shape);
    test5.copyfrom(test1);
    gpu_fftshift(test5, 0, FORWARD);
    CHECK_EQ(test5.data_at({0, 1, 2}), 23.0);
    CHECK_EQ(test5.data_at({1, 1, 2}), 5.0);
    CHECK_EQ(test5.data_at({2, 1, 2}), 14.0);

    Tensor<float> test6(shape);
    test6.copyfrom(test1);
    gpu_fftshift(test6, 1, FORWARD);
    CHECK_EQ(test6.data_at({0, 1, 2}), 2.0);
    CHECK_EQ(test6.data_at({1, 1, 2}), 11.0);
    CHECK_EQ(test6.data_at({2, 1, 2}), 20.0);

    Tensor<float> test7(shape);
    test7.copyfrom(test1);
    gpu_fftshift(test7, 2, FORWARD);
    CHECK_EQ(test7.data_at({0, 1, 2}), 4.0);
    CHECK_EQ(test7.data_at({1, 1, 2}), 13.0);
    CHECK_EQ(test7.data_at({2, 1, 2}), 22.0);

    // test gpu_fft
    const int N = 194910;
    cuFloatComplex *host_val = (cuFloatComplex *)malloc(N * sizeof(cuFloatComplex));
    for (int i = 0; i < N; i++)
    {
        host_val[i].x = (float)rand() / (float)RAND_MAX;
        host_val[i].y = (float)rand() / (float)RAND_MAX;
    }
    Tensor<cuFloatComplex> test8({N});
    Tensor<cuFloatComplex> test9({N});
    test8.set_cpu_data(host_val);
    test9.copyfrom(test8);
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    gpu_fftshift(test8, 0, INVERSE);
    CUFFT_CHECK(cufftExecC2C(plan, test8.mutable_gpu_data(),
                             test8.mutable_gpu_data(), CUFFT_INVERSE));
    gpu_fftshift(test8, 0, INVERSE);
    gpu_normalizing(test8, N);
    gpu_fft(test9, {0}, INVERSE, SHIFT);
    for (int i = 0; i < N; i++)
    {
        CHECK_EQ(test8.data_at({i}) == test9.data_at({i}), true);
    }

    cuFloatComplex *data_ = (cuFloatComplex *)malloc(8000 * sizeof(cuFloatComplex));
    for (int i = 0; i < 8000; ++i)
        data_[i] = make_cuFloatComplex((float)i, (float)i);
    Tensor<cuFloatComplex> test10({20, 20, 20});
    test10.set_cpu_data(data_);
    gpu_fft(test10, {0, 1}, INVERSE, NON_SHIFT);
    CHECK_EQ(test10.data_at({0, 0, 0}) == make_cuFloatComplex(3990.0, 3990.0), true);
    CHECK_EQ(test10.data_at({0, 1, 1}) ==
                 make_cuFloatComplex(53.13751514675044, -73.13751514675043),
             true);
    CHECK_EQ(test10.data_at({1, 1, 1}) == make_cuFloatComplex(0.0, 0.0), true);
    CHECK_EQ(test10.data_at({7, 11, 13}) == make_cuFloatComplex(0.0, 0.0), true);
    CHECK_EQ(test10.data_at({12, 3, 5}) ==
                 make_cuFloatComplex(-1.7763568394002506e-16, -7.105427357601002e-16),
             true);
    CHECK_EQ(test10.data_at({16, 19, 1}) ==
                 make_cuFloatComplex(-2.842170943040401e-15, 0.0),
             true);
    CHECK_EQ(test10.data_at({18, 0, 1}) ==
                 make_cuFloatComplex(-815.5367074350506, 415.53670743505063),
             true);
    CHECK_EQ(test10.data_at({19, 13, 11}) == make_cuFloatComplex(0.0, 0.0), true);

    // test gpu_dot
    /* ----------------------------------------------------------------
 * Set test2 as below:
 * 1.0 + 1.0j, 7.0 - 7.0j, 0.0,        0.0
 * 0.0,        2.0 + 2.0j, 8.0 - 8.0j, 0.0
 * 5.0 + 5.0j, 0.0,        3.0 - 3.0j, 9.0 + 9.0j
 * 0.0,        6.0 - 6.0j, 0.0,        4.0 + 4.0j
 * ----------------------------------------------------------------*/
    int *ptr = new int[5];
    int *indices = new int[9];
    data_ = new cuFloatComplex[9];
    data_[0] = make_cuFloatComplex(1.0, +1.0), indices[0] = 0, ptr[0] = 0;
    data_[1] = make_cuFloatComplex(7.0, -7.0), indices[1] = 1, ptr[1] = 2;
    data_[2] = make_cuFloatComplex(2.0, +2.0), indices[2] = 1, ptr[2] = 4;
    data_[3] = make_cuFloatComplex(8.0, -8.0), indices[3] = 2, ptr[3] = 7;
    data_[4] = make_cuFloatComplex(5.0, +5.0), indices[4] = 0, ptr[4] = 9;
    data_[5] = make_cuFloatComplex(3.0, -3.0), indices[5] = 2;
    data_[6] = make_cuFloatComplex(9.0, +9.0), indices[6] = 3;
    data_[7] = make_cuFloatComplex(6.0, -6.0), indices[7] = 1;
    data_[8] = make_cuFloatComplex(4.0, +4.0), indices[8] = 3;

    SparseTensor<cuFloatComplex> test11({4, 4}, 9);
    test11.set_cpu_data(data_, indices, ptr);
    Tensor<cuFloatComplex> test12({4, 4});
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseCcsr2dense(GPU_Controller::cuSPARSE_handle(),
                                      4, 4, descr, test11.gpu_data(), test11.gpu_ptr(),
                                      test11.gpu_indices(),
                                      test12.mutable_gpu_data(), 4));
    test12.swapaxes(0, 1);
    Tensor<cuFloatComplex> test13;
    gpu_dot(test11, test12, test13);
    CHECK_EQ(test13.data_at({0, 0}) == make_cuFloatComplex(0.0, 2.0), true);
    CHECK_EQ(test13.data_at({0, 1}) == make_cuFloatComplex(42.0, 0.0), true);
    CHECK_EQ(test13.data_at({0, 2}) == make_cuFloatComplex(0.0, -112.0), true);
    CHECK_EQ(test13.data_at({0, 3}) == make_cuFloatComplex(0.0, 0.0), true);
    CHECK_EQ(test13.data_at({1, 0}) == make_cuFloatComplex(80.0, 0.0), true);
    CHECK_EQ(test13.data_at({1, 1}) == make_cuFloatComplex(0.0, 8.0), true);
    CHECK_EQ(test13.data_at({1, 2}) == make_cuFloatComplex(32.0, -48.0), true);
    CHECK_EQ(test13.data_at({1, 3}) == make_cuFloatComplex(144.0, 0.0), true);
    CHECK_EQ(test13.data_at({2, 0}) == make_cuFloatComplex(30.0, 10.0), true);
    CHECK_EQ(test13.data_at({2, 1}) == make_cuFloatComplex(178.0, 0.0), true);
    CHECK_EQ(test13.data_at({2, 2}) == make_cuFloatComplex(0.0, -18.0), true);
    CHECK_EQ(test13.data_at({2, 3}) == make_cuFloatComplex(54.0, 72.0), true);
    CHECK_EQ(test13.data_at({3, 0}) == make_cuFloatComplex(0.0, 0.0), true);
    CHECK_EQ(test13.data_at({3, 1}) == make_cuFloatComplex(72.0, 0.0), true);
    CHECK_EQ(test13.data_at({3, 2}) == make_cuFloatComplex(0.0, -96.0), true);
    CHECK_EQ(test13.data_at({3, 3}) == make_cuFloatComplex(0.0, 32.0), true);

    printf("Testing passed.\n");

    return 0;
}