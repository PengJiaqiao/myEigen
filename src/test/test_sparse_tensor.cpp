// nvcc -std=c++11 test_sparse_tensor.cpp ../sparse_tensor.cpp ../tensor.cpp ../permute.cu ../syncedmem.cpp ../common.cpp -I ../../include -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L ../../third_party -lglog -lgflags -lpthread -lcuda -lcusparse -lcufft -lcudart -o test_sparse_tensor

#include "sparse_tensor.hpp"
#include "tensor.hpp"

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

    vector<int> shape = {3, 3};
    int nnz = 6;
    SparseTensor<float> test1(shape, nnz);
    CHECK_EQ(test1.shape().size(), 2);
    CHECK_EQ(test1.shape()[0], 3);
    CHECK_EQ(test1.shape()[1], 3);
    CHECK_EQ(test1.num_axes(), 2);
    CHECK_EQ(test1.nnz(), 6);
    CHECK_EQ(test1.row(), 3);

    SparseTensor<float> test2;
    test2.copyfrom(test1);
    CHECK_EQ(test2.shape().size(), 2);
    CHECK_EQ(test2.shape()[0], 3);
    CHECK_EQ(test2.shape()[1], 3);
    CHECK_EQ(test2.num_axes(), 2);
    CHECK_EQ(test2.nnz(), 6);
    CHECK_EQ(test2.row(), 3);

    vector<int> new_shape = {4, 4};
    nnz = 9;
    test1.reshape(new_shape, nnz);
    CHECK_EQ(test1.shape().size(), 2);
    CHECK_EQ(test1.shape()[0], 4);
    CHECK_EQ(test1.shape()[1], 4);
    CHECK_EQ(test1.num_axes(), 2);
    CHECK_EQ(test1.nnz(), 9);
    CHECK_EQ(test1.row(), 4);

    test2.reshapelike(test1);
    CHECK_EQ(test2.shape().size(), 2);
    CHECK_EQ(test2.shape()[0], 4);
    CHECK_EQ(test2.shape()[1], 4);
    CHECK_EQ(test2.num_axes(), 2);
    CHECK_EQ(test2.nnz(), 9);
    CHECK_EQ(test2.row(), 4);

    /* ----------------------------------------------------------------
 * Set test2 as below:
 * 1.0 7.0 0.0 0.0
 * 0.0 2.0 8.0 0.0
 * 5.0 0.0 3.0 9.0
 * 0.0 6.0 0.0 4.0
 * ----------------------------------------------------------------*/
    int *ptr = new int[5];
    int *indices = new int[9];
    float *data = new float[9];
    data[0] = 1.0, indices[0] = 0, ptr[0] = 0;
    data[1] = 7.0, indices[1] = 1, ptr[1] = 2;
    data[2] = 2.0, indices[2] = 1, ptr[2] = 4;
    data[3] = 8.0, indices[3] = 2, ptr[3] = 7;
    data[4] = 5.0, indices[4] = 0, ptr[4] = 9;
    data[5] = 3.0, indices[5] = 2;
    data[6] = 9.0, indices[6] = 3;
    data[7] = 6.0, indices[7] = 1;
    data[8] = 4.0, indices[8] = 3;

    test2.set_cpu_data(data, indices, ptr);
    Tensor<float> test3(new_shape);
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseScsr2dense(GPU_Controller::cuSPARSE_handle(),
                                      4, 4, descr, test2.gpu_data(), test2.gpu_ptr(),
                                      test2.gpu_indices(),
                                      test3.mutable_gpu_data(), 4));
    test3.swapaxes(0, 1);
    CHECK_EQ(test3.data_at({0, 0}), 1.0);
    CHECK_EQ(test3.data_at({0, 1}), 7.0);
    CHECK_EQ(test3.data_at({0, 2}), 0.0);
    CHECK_EQ(test3.data_at({0, 3}), 0.0);
    CHECK_EQ(test3.data_at({1, 0}), 0.0);
    CHECK_EQ(test3.data_at({1, 1}), 2.0);
    CHECK_EQ(test3.data_at({1, 2}), 8.0);
    CHECK_EQ(test3.data_at({1, 3}), 0.0);
    CHECK_EQ(test3.data_at({2, 0}), 5.0);
    CHECK_EQ(test3.data_at({2, 1}), 0.0);
    CHECK_EQ(test3.data_at({2, 2}), 3.0);
    CHECK_EQ(test3.data_at({2, 3}), 9.0);
    CHECK_EQ(test3.data_at({3, 0}), 0.0);
    CHECK_EQ(test3.data_at({3, 1}), 6.0);
    CHECK_EQ(test3.data_at({3, 2}), 0.0);
    CHECK_EQ(test3.data_at({3, 3}), 4.0);

    printf("Testing passed.\n");

    return 0;
}