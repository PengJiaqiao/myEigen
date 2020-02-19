#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <algorithm>
#include <string>
#include <vector>
#include <sstream>

#include "common.hpp"
#include "syncedmem.hpp"

namespace MRI
{

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit.
 */
template <typename Dtype>
class Tensor
{
public:
	Tensor() : data_(), count_(0), capacity_(0) {}

	explicit Tensor(const vector<int> &shape);

	// Change the dimensions of the Tensor, allocating new memory if necessary.
	void reshape(const vector<int> &shape);
	void reshapelike(const Tensor &other);

	// Print each dimension
	inline string shape_string() const
	{
		ostringstream stream;
		for (int i = 0; i < shape_.size(); ++i)
		{
			stream << shape_[i] << " ";
		}
		stream << "(" << count_ << ")";
		return stream.str();
	}

	inline const vector<int> &shape() const { return shape_; }

	// Returns the dimension of the index-th axis (or the negative
	// index-th axis from the end, if index is negative).
	inline int shape(int index) const
	{
		return shape_[CanonicalAxisIndex(index)];
	}
	inline int num_axes() const { return shape_.size(); }
	inline int count() const { return count_; }
	inline int capacity() const { return capacity_; }

	/**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
	inline int count(int start_axis, int end_axis) const
	{
		CHECK_LE(start_axis, end_axis);
		CHECK_GE(start_axis, 0);
		CHECK_GE(end_axis, 0);
		CHECK_LE(start_axis, num_axes());
		CHECK_LE(end_axis, num_axes());
		int count = 1;
		for (int i = start_axis; i < end_axis; ++i)
		{
			count *= shape(i);
		}
		return count;
	}
	/**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
	inline int count(int start_axis) const
	{
		return count(start_axis, num_axes());
	}

	/**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
	inline int CanonicalAxisIndex(int axis_index) const
	{
		CHECK_GE(axis_index, -num_axes())
			<< "axis " << axis_index << " out of range for " << num_axes()
			<< "-D Tensor with shape " << shape_string();
		CHECK_LT(axis_index, num_axes())
			<< "axis " << axis_index << " out of range for " << num_axes()
			<< "-D Tensor with shape " << shape_string();
		if (axis_index < 0)
		{
			return axis_index + num_axes();
		}
		return axis_index;
	}

	/**
   * @brief Copy from a source Tensor.
   *
   * @param source the Tensor to copy from
   * @param reshape if false, require this Tensor to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Tensor to other's
   *        shape if necessary
   */
	void copyfrom(const Tensor<Dtype> &source, bool reshape = true);

	inline int offset(const vector<int> &indices) const
	{
		CHECK_LE(indices.size(), num_axes());
		int offset = 0;
		for (int i = 0; i < num_axes(); ++i)
		{
			offset *= shape(i);
			if (indices.size() > i)
			{
				CHECK_GE(indices[i], 0);
				CHECK_LT(indices[i], shape(i));
				offset += indices[i];
			}
		}
		return offset;
	}

	inline Dtype data_at(const vector<int> &index) const
	{
		return cpu_data()[offset(index)];
	}

	inline const shared_ptr<SyncedMemory> &data() const
	{
		CHECK(data_);
		return data_;
	}

	const Dtype *cpu_data() const;
	void set_cpu_data(Dtype *data);
	const Dtype *gpu_data() const;
	void set_gpu_data(Dtype *data);
	Dtype *mutable_cpu_data(); // read and write
	Dtype *mutable_gpu_data(); // read and write

	// TODO: data proto
	// void FromProto(const TensorProto &proto, bool reshape = true);
	// void ToProto(TensorProto *proto, bool write_diff = false) const;

	// TODO:
	/**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_.
   *
   */
	// void ShareData(const Tensor &other);

	/**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Tensor other.
   *
   * This deallocates the SyncedMemory holding this Tensor's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
	void sharedata(const Tensor &other);

	// The same as numpy.ndarray.transpose()
	void transpose(const vector<int> &permute_order);

	// The same as numpy.ndarray.swapaxes()
	void swapaxes(const int &axis1, const int &axis2);

protected:
	shared_ptr<SyncedMemory> data_;
	vector<int> shape_;
	int count_;
	int capacity_;

	DISABLE_COPY_AND_ASSIGN(Tensor);
}; // class Tensor

} // namespace MRI

#endif // TENSOR_HPP_
