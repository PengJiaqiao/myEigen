#include <vector>

#include "common.hpp"
#include "sparse_tensor.hpp"
#include "syncedmem.hpp"

namespace MRI
{

template <typename Dtype>
void SparseTensor<Dtype>::reshape(const vector<int> &shape, const int nnz)
{
    CHECK_EQ(shape.size(), 2);
    CHECK_GT(shape[0], 0);
    CHECK_GT(shape[1], 0);
    CHECK_GE(nnz, 0);

    int previous_num = 0;
    if (shape_.size() > 0)
    {
        previous_num = shape_[0];
    }
    shape_.resize(2);
    shape_[0] = shape[0];
    shape_[1] = shape[1];
    count_ = shape[0] * shape[1];
    if (count_)
    {
        if (nnz != nnz_)
        {
            nnz_ = nnz;
            data_.reset(new SyncedMemory(nnz_ * sizeof(Dtype)));
            indices_.reset(new SyncedMemory(nnz_ * sizeof(int)));
        }
        if (previous_num != shape[0])
        {
            ptr_.reset(new SyncedMemory((shape_[0] + 1) * sizeof(int)));
        }
    }
    else
    {
        data_.reset(static_cast<SyncedMemory *>(nullptr));
        indices_.reset(static_cast<SyncedMemory *>(nullptr));
        ptr_.reset(static_cast<SyncedMemory *>(nullptr));
    }
}

template <typename Dtype>
void SparseTensor<Dtype>::reshape(const int row, const int column, const int nnz)
{
    CHECK_GT(row, 0);
    CHECK_GT(column, 0);
    vector<int> shape(2);
    shape[0] = row;
    shape[1] = column;
    reshape(shape, nnz);
}

template <typename Dtype>
void SparseTensor<Dtype>::reshapelike(const SparseTensor<Dtype> &other)
{
    reshape(other.shape(), other.nnz());
}

template <typename Dtype>
SparseTensor<Dtype>::SparseTensor(const vector<int> &shape,
                                  const int nnz)
    : nnz_(0)
{
    reshape(shape, nnz);
}

template <typename Dtype>
SparseTensor<Dtype>::SparseTensor(const int row, const int column, const int nnz)
    : nnz_(0)
{
    vector<int> shape(2);
    shape[0] = row;
    shape[1] = column;
    reshape(shape, nnz);
}

template <typename Dtype>
void MRI_copy(const int N, const Dtype *src, Dtype *dst)
{
    if (src != dst)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, sizeof(Dtype) * N, cudaMemcpyDefault));
    }
}

template <typename Dtype>
void SparseTensor<Dtype>::copyfrom(const SparseTensor<Dtype> &source, bool reshape)
{
    if (source.count() != count_ || source.shape() != shape_)
    {
        if (reshape)
        {
            reshapelike(source);
        }
        else
        {
            LOG(FATAL) << "Trying to copy Sparse Tensors of different sizes.";
        }
    }
    MRI_copy(nnz_, source.gpu_data(), static_cast<Dtype *>(data_->mutable_gpu_data()));
    MRI_copy(nnz_, source.gpu_indices(), static_cast<int *>(indices_->mutable_gpu_data()));
    MRI_copy(row() + 1, source.gpu_ptr(), static_cast<int *>(ptr_->mutable_gpu_data()));
}

template <typename Dtype>
void SparseTensor<Dtype>::set_cpu_data(Dtype *data, int *indices, int *ptr)
{
    CHECK(data);
    CHECK(indices);
    CHECK(ptr);
    data_->set_cpu_data(static_cast<void *>(data));
    indices_->set_cpu_data(static_cast<void *>(indices));
    ptr_->set_cpu_data(static_cast<void *>(ptr));
}

template <typename Dtype>
void SparseTensor<Dtype>::set_gpu_data(Dtype *data, int *indices, int *ptr)
{
    CHECK(data);
    CHECK(indices);
    CHECK(ptr);
    data_->set_gpu_data(static_cast<void *>(data));
    indices_->set_gpu_data(static_cast<void *>(indices));
    ptr_->set_gpu_data(static_cast<void *>(ptr));
}

template <typename Dtype>
const Dtype *SparseTensor<Dtype>::cpu_data() const
{
    CHECK(data_);
    return (const Dtype *)data_->cpu_data();
}

template <typename Dtype>
const int *SparseTensor<Dtype>::cpu_indices() const
{
    CHECK(indices_);
    return (const int *)indices_->cpu_data();
}

template <typename Dtype>
const int *SparseTensor<Dtype>::cpu_ptr() const
{
    CHECK(ptr_);
    return (const int *)ptr_->cpu_data();
}

template <typename Dtype>
const Dtype *SparseTensor<Dtype>::gpu_data() const
{
    CHECK(data_);
    return (const Dtype *)data_->gpu_data();
}

template <typename Dtype>
const int *SparseTensor<Dtype>::gpu_indices() const
{
    CHECK(indices_);
    return (const int *)indices_->gpu_data();
}

template <typename Dtype>
const int *SparseTensor<Dtype>::gpu_ptr() const
{
    CHECK(ptr_);
    return (const int *)ptr_->gpu_data();
}

template <typename Dtype>
Dtype *SparseTensor<Dtype>::mutable_cpu_data()
{
    CHECK(data_);
    return static_cast<Dtype *>(data_->mutable_cpu_data());
}

template <typename Dtype>
int *SparseTensor<Dtype>::mutable_cpu_indices()
{
    CHECK(indices_);
    return static_cast<int *>(indices_->mutable_cpu_data());
}

template <typename Dtype>
int *SparseTensor<Dtype>::mutable_cpu_ptr()
{
    CHECK(ptr_);
    return static_cast<int *>(ptr_->mutable_cpu_data());
}

template <typename Dtype>
Dtype *SparseTensor<Dtype>::mutable_gpu_data()
{
    CHECK(data_);
    return static_cast<Dtype *>(data_->mutable_gpu_data());
}

template <typename Dtype>
int *SparseTensor<Dtype>::mutable_gpu_indices()
{
    CHECK(indices_);
    return static_cast<int *>(indices_->mutable_gpu_data());
}

template <typename Dtype>
int *SparseTensor<Dtype>::mutable_gpu_ptr()
{
    CHECK(ptr_);
    return static_cast<int *>(ptr_->mutable_gpu_data());
}

INSTANTIATE_CLASS(SparseTensor);

} // namespace MRI
