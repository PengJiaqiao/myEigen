#ifndef SPARSE_TENSOR_HPP_
#define SPARSE_TENSOR_HPP_

#include <vector>

#include "common.hpp"
#include "syncedmem.hpp"

namespace MRI
{

/**
 * @brief  An extension of Tensor to support sparse matrix in CSR format
 * TODO: making SparseTensor a child class of Tensor and override the methods
 *
 */
template <typename Dtype>
class SparseTensor
{
public:
    SparseTensor()
        : data_(), count_(0), indices_(), ptr_(), nnz_(0) {}

    explicit SparseTensor(const vector<int> &shape, const int nnz);
    explicit SparseTensor(const int row, const int column, const int nnz);

    ~SparseTensor() {}

    // Change the dimensions of the SparseTensor, allocating new memory if necessary.
    void reshape(const vector<int> &shape, const int nnz);
    void reshape(const int row, const int column, const int nnz);
    void reshapelike(const SparseTensor<Dtype> &other);

    inline const vector<int> &shape() const { return shape_; }
    inline int num_axes() const { return shape_.size(); }
    inline int count() const { return count_; }
    inline int nnz() const { return nnz_; }
    inline int row() const { return shape_[0]; }
    inline int col() const { return shape_[1]; }

    void copyfrom(const SparseTensor<Dtype> &source, bool reshape = true);

    inline int offset(const vector<int> &index) const
    {
        NOT_IMPLEMENTED;
        return 0;
    }
    inline int offset(const int row, const int column) const
    {
        NOT_IMPLEMENTED;
        return 0;
    }
    /*
    inline Dtype data_at(const vector<int> &index) const
    {
        NOT_IMPLEMENTED;
        return 0;
    }
    inline Dtype data_at(const int row, const int column) const
    {
        NOT_IMPLEMENTED;
        return 0;
    }*/

    inline const shared_ptr<SyncedMemory> &data() const
    {
        CHECK(data_);
        return data_;
    }
    inline const shared_ptr<SyncedMemory> &indices() const
    {
        CHECK(indices_);
        return indices_;
    }
    inline const shared_ptr<SyncedMemory> &ptr() const
    {
        CHECK(ptr_);
        return ptr_;
    }

    const Dtype *cpu_data() const;
    const int *cpu_indices() const;
    const int *cpu_ptr() const;
    const Dtype *gpu_data() const;
    const int *gpu_indices() const;
    const int *gpu_ptr() const;

    Dtype *mutable_cpu_data(); // read and write
    int *mutable_cpu_indices();
    int *mutable_cpu_ptr();
    Dtype *mutable_gpu_data(); // read and write
    int *mutable_gpu_indices();
    int *mutable_gpu_ptr();

    void set_cpu_data(Dtype *data, int *indices, int *ptr);
    void set_gpu_data(Dtype *data, int *indices, int *ptr);

    /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Tensor other.
   *
   * This deallocates the SyncedMemory holding this Tensor's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
    void sharedata(const SparseTensor<Dtype> &other);

protected:
    shared_ptr<SyncedMemory> data_;
    shared_ptr<SyncedMemory> indices_; // CSR format column indices
    shared_ptr<SyncedMemory> ptr_;     // CSR format row offsets
    vector<int> shape_;
    int count_;
    int nnz_;

    DISABLE_COPY_AND_ASSIGN(SparseTensor);

}; // class SparseTensor

} // namespace MRI

#endif // SPARSE_TENSOR_HPP_