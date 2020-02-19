#include <vector>

#include "tensor.hpp"
#include "common.hpp"
#include "syncedmem.hpp"
#include "permute.hpp"

namespace MRI
{

template <typename Dtype>
void Tensor<Dtype>::reshape(const vector<int> &shape)
{
	if (shape.size() == 0)
	{
		count_ = 0;
	}
	else
	{
		count_ = 1;
		shape_.resize(shape.size());
		for (int i = 0; i < shape.size(); ++i)
		{
			CHECK_GE(shape[i], 0);
			if (count_ != 0)
			{
				CHECK_LE(shape[i], INT_MAX / count_) << "Tensor size exceeds INT_MAX";
			}
			count_ *= shape[i];
			shape_[i] = shape[i];
		}
		if (count_ > capacity_)
		{
			capacity_ = count_;
			data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
		}
	}
}

template <typename Dtype>
void Tensor<Dtype>::reshapelike(const Tensor<Dtype> &other)
{
	reshape(other.shape());
}

template <typename Dtype>
Tensor<Dtype>::Tensor(const vector<int> &shape)
	// capacity_ must be initialized before calling Reshape
	: capacity_(0)
{
	reshape(shape);
}

template <typename Dtype>
const Dtype *Tensor<Dtype>::cpu_data() const
{
	CHECK(data_);
	return (const Dtype *)data_->cpu_data();
}

template <typename Dtype>
void Tensor<Dtype>::set_cpu_data(Dtype *data)
{
	CHECK(data);
	// Make sure CPU and GPU sizes remain equal
	size_t size = count_ * sizeof(Dtype);
	if (data_->size() != size)
	{
		data_.reset(new SyncedMemory(size));
	}
	data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype *Tensor<Dtype>::gpu_data() const
{
	CHECK(data_);
	return (const Dtype *)data_->gpu_data();
}

template <typename Dtype>
void Tensor<Dtype>::set_gpu_data(Dtype *data)
{
	CHECK(data);
	// Make sure CPU and GPU sizes remain equal
	size_t size = count_ * sizeof(Dtype);
	if (data_->size() != size)
	{
		data_.reset(new SyncedMemory(size));
	}
	data_->set_gpu_data(data);
}

template <typename Dtype>
Dtype *Tensor<Dtype>::mutable_cpu_data()
{
	CHECK(data_);
	return static_cast<Dtype *>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype *Tensor<Dtype>::mutable_gpu_data()
{
	CHECK(data_);
	return static_cast<Dtype *>(data_->mutable_gpu_data());
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
void Tensor<Dtype>::copyfrom(const Tensor &source, bool reshape)
{
	if (source.count() != count_ || source.shape() != shape_)
	{
		if (reshape)
		{
			reshapelike(source);
		}
		else
		{
			LOG(FATAL) << "Trying to copy Tensors of different sizes.";
		}
	}
	MRI_copy(count_, source.gpu_data(), static_cast<Dtype *>(data_->mutable_gpu_data()));
}

template <typename Dtype>
void Tensor<Dtype>::sharedata(const Tensor &other)
{
	data_ = other.data();
	shape_ = other.shape();
	count_ = other.count();
	capacity_ = other.capacity();
}

template <typename Dtype>
void Tensor<Dtype>::transpose(const vector<int> &permute_order)
{
	MRI::transpose(*this, permute_order);
}

template <typename Dtype>
void Tensor<Dtype>::swapaxes(const int &axis1, const int &axis2)
{
	MRI::swapaxes(*this, axis1, axis2);
}

INSTANTIATE_CLASS(Tensor);

} // namespace MRI
