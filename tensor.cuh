#ifndef MYTORCH_H_
#define MYTORCH_H_

#include <memory>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

#include "curand.h"


namespace mytorch{
    namespace nn{
        namespace Functional{
            template< typename T>
            class Function;
            template<typename T>
            class AddFunc;
            template<typename T>
            class NegFunc;
            template<typename T>
            class SubFunc;
            template<typename T>
            class MulFunc;
            template<typename T>
            class DivFunc;
            template<typename T>
            class ReLUFunc;
            template<typename T>
            class SigmoidFunc;
            template<typename T>
            class TransposeFunc;
            template<typename T>
            class ReshapeFunc;
            template<typename T>
            class MatmulFunc;
        }
    };

    const size_t kCudaThreadsNum = 512;
    inline int CudaGetBlocks(const int N) {
        return(N + kCudaThreadsNum-1) / kCudaThreadsNum;
    }
    // Define the grid stride looping
    #define CUDA_KERNEL_LOOP(i, n)                         \
        for(int i =blockIdx.x*blockDim.x+threadIdx.x;  \
            i<(n);                                        \
            i+= blockDim.x * gridDim.x)
    // Check for CUDA errors
    #define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (line: " << __LINE__ << ") infile: " << __FILE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
    #define CHECK_CURAND(call) \
    do { \
        curandStatus_t err = call; \
        if (err != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "cuRAND错误: 代码 %d (行号: %d)\n", err, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

    enum Device{
        Cpu , Cuda
    };

    /* cuda_shared_pointer for memory management */

    template <typename T>
    class cuda_shared_pointer{
        private:
            T * data_;
            Device device_;
            size_t size_;
            std::shared_ptr<int> ref_count_;
            void allocate( size_t size , Device device){
                size_ = size;
                device_ = device;
                ref_count_ = std::make_shared<int>(1);
                if (device_ == Cpu){
                    data_ = new T[size_];
                }
                else{
                    CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
                    CHECK(cudaMemset(data_ , 0 , size_ * sizeof(T)));
                }
            }
            void release(){
                if (ref_count_ && --(*ref_count_) == 0){
                    if (device_ == Cpu){
                        delete[] data_;
                    }
                    else{
                        CHECK(cudaFree(data_));
                    }
                }
            }
        public:
            int ref_count() const{
                return ref_count_ ? *ref_count_ : 0;
            }
            cuda_shared_pointer(size_t size , Device device = Cpu){
                allocate(size , device);
            }
            cuda_shared_pointer(const T * data ,const size_t size ,const Device device = Cpu){
                if(device == Cpu){
                    size_ = size;
                    device_ = Cpu;
                    ref_count_ = std::make_shared<int>(1);
                    memcpy(data_ , data , size_ * sizeof(T));
                }
                else{
                    size_ = size;
                    device_ = Cuda;
                    ref_count_ = std::make_shared<int>(1);
                    CHECK(cudaMemcpy(data_ , data , size_ * sizeof(T) , cudaMemcpyHostToDevice));
                }
            }
            cuda_shared_pointer(){
                data_ = nullptr;
                device_ = Cpu;
                size_ = 0;
                ref_count_ = nullptr;
            }
            ~cuda_shared_pointer(){
                release();
            }
            T * get(){
                return data_;
            }
            const T * get() const{
                return data_;
            }
            Device device() const{
                return device_;
            }
            size_t size() const{
                return size_;
            }
            bool is_null() const{
                return data_ == nullptr;
            }
            void to(Device device){
                if (device_ == device) {
                    return;
                }
                if (data_ == nullptr || size_ == 0){
                    device_ = device;
                    return ;
                }
                
                if(device == Cpu){
                    T * cpu_data = new T[size_];
                    CHECK(cudaMemcpy(cpu_data , data_ , size_ * sizeof(T) , cudaMemcpyDeviceToHost));
                    CHECK(cudaFree(data_));
                    data_ = cpu_data;
                }
                else{
                    T * cuda_data = nullptr;
                    CHECK(cudaMalloc(&cuda_data, size_ * sizeof(T)));
                    CHECK(cudaMemcpy(cuda_data , data_ , size_ * sizeof(T) , cudaMemcpyHostToDevice));
                    delete[] data_;
                    data_ = cuda_data;
                }
                device_ = device;
            }
            cuda_shared_pointer(const cuda_shared_pointer& other){
                data_ = other.data_;
                device_ = other.device_;
                size_ = other.size_;
                ref_count_ = other.ref_count_;
                (*ref_count_)++;
            }
            cuda_shared_pointer& operator=(const cuda_shared_pointer& other){
                if(this != &other){
                    release();
                    data_ = other.data_;
                    device_ = other.device_;
                    size_ = other.size_;
                    ref_count_ = other.ref_count_;
                    if (ref_count_){
                        (*ref_count_)++;
                    }
                }
                return *this;
            }
            T& operator[](size_t index){
                if (index >= size_){
                    throw std::runtime_error("Index out of range!");
                }
                if(device_ != Device::Cpu){
                    throw std::runtime_error("Cannot dereference on cuda!");
                }
                return data_[index];
            }
            const T& operator[](size_t index) const{
                if(index >= size_){
                    throw std::runtime_error("Index out of range!");
                }
                if(device_ != Device::Cpu){
                    throw std::runtime_error("Cannot dereference on cuda!");
                }
                return data_[index];
            }
            operator T*() const {
                return data_;
            }
            cuda_shared_pointer<T> deepcopy() const{
                cuda_shared_pointer<T> result;
                result.size_ = size_;
                result.device_ = device_;
                result.ref_count_ = std::make_shared<int>(1);
                if (device_ == Cpu){
                    result.data_ = new T[size_];
                    memcpy(result.data_ , data_ , size_ * sizeof(T));
                }
                else{
                    CHECK(cudaMalloc(&result.data_, result.size_ * sizeof(T)));
                    CHECK(cudaMemcpy(result.data_ , data_ , result.size_ * sizeof(T) , cudaMemcpyDeviceToDevice));
                }

                return result;
            }
            cuda_shared_pointer(const std::vector<T> & data , const Device device = Cpu){
                size_ = data.size();
                device_ = device;
                ref_count_ = std::make_shared<int>(1);
                if (device_ == Cpu){
                    data_ = new T[size_];
                    memcpy(data_ , data.data() , size_ * sizeof(T));
                }
                else{
                    CHECK(cudaMalloc(&data_ , size_ * sizeof(T)));
                    CHECK(cudaMemcpy(data_ , data.data() , size_ * sizeof(T) , cudaMemcpyHostToDevice));
                }
            }
    };





    template <typename T>
    class TensorRaw{
        private:
            size_t size_;
            std::vector<size_t> shape_;
            std::vector<size_t> strides;
            cuda_shared_pointer<T> data_;
            bool requires_grad_;
            cuda_shared_pointer<T> grad_;
            std::shared_ptr<nn::Functional::Function<T>> grad_fn_;
            size_t get_strides_with_shape(const std::vector<size_t> & shape){
                strides = {};
                size_t nowstride = 1;
                for(int i = shape.size() - 1;i>= 0;i--){
                    strides.push_back(nowstride);
                    nowstride *= shape[i];
                }
                return nowstride;
            }
        public:
            friend class nn::Functional::AddFunc<T>;
            friend class nn::Functional::Function<T>;
            friend class nn::Functional::NegFunc<T>;
            friend class nn::Functional::SubFunc<T>;
            friend class nn::Functional::MulFunc<T>;
            friend class nn::Functional::DivFunc<T>;
            friend class nn::Functional::ReLUFunc<T>;
            friend class nn::Functional::SigmoidFunc<T>;
            friend class nn::Functional::TransposeFunc<T>;
            friend class nn::Functional::ReshapeFunc<T>;
            friend class nn::Functional::MatmulFunc<T>;

            TensorRaw<T>(const T * data , const std::vector<size_t> & shape , const Device device = Cpu){
                size_ = get_strides_with_shape(shape);
                shape_ = shape;
                data_ = cuda_shared_pointer<T>(data , size_ , device);
                requires_grad_ = false;
            }
            TensorRaw<T>(const T value , const std::vector<size_t> & shape = {1} , const Device device = Cpu){
                size_ = get_strides_with_shape(shape);
                shape_ = shape;
                data_ = cuda_shared_pointer<T>(value , size_ , device);
                requires_grad_ = false;
            }
            size_t size() const{
                return size_;
            }
            void set_requires_grad(bool requires_grad){
                requires_grad_ = requires_grad;
            }
            bool requires_grad() const {
                return requires_grad_;
            }
            void set_grad_fn(const std::shared_ptr<nn::Functional::Function<T>> & grad_fn){
                grad_fn_ = grad_fn;
            }
            std::shared_ptr<nn::Functional::Function<T>> get_grad_fn() const{
                return grad_fn_;
            }
            void set_grad(const cuda_shared_pointer<T> & grad){
                grad_ = grad;
            }
            const cuda_shared_pointer<T> & get_grad() const{
                return grad_;
            }
            cuda_shared_pointer<T> & get_shared_ptr(){
                return data_;
            }
            const cuda_shared_pointer<T> & get_shared_ptr() const{
                return data_;
            }

            TensorRaw(const TensorRaw& other) {
                shape_ = other.shape_;
                strides = other.strides;
                size_ = other.size_;
                data_ = other.data_;
                requires_grad_ = other.requires_grad_;
                grad_ = other.grad_;
                grad_fn_ = other.grad_fn_;
            }
            TensorRaw & operator=(const TensorRaw& other) {
                if (this == &other) return *this;
                shape_ = other.shape_;
                strides = other.strides;
                size_ = other.size_;
                data_ = other.data_;
                requires_grad_ = other.requires_grad_;
                grad_ = other.grad_;
                grad_fn_ = other.grad_fn_;
                return *this;
            }
            Device device() const{
                return data_.device();
            }
            const std::vector<size_t> & shape() const{
                return shape_;
            }
            void to(Device device){
                data_.to(device);
            }

            T * get(){
                return data_.get();
            }
            const T * get() const{
                return data_.get();
            }
            std::vector<size_t> get_strides() const{
                return strides;
            }

            
            TensorRaw(const std::vector<size_t> & shape ,const Device device=Cpu){
                shape_ = shape;
                size_ = get_strides_with_shape(shape);
                data_ = cuda_shared_pointer<T>(size_ , device);
                if (device == Cpu){
                    std::fill(data_.get(), data_.get() + size_, T(0));
                }
                else{
                    CHECK(cudaMemset(data_.get(), 0, size_ * sizeof(T)));
                }
                requires_grad_ = false;
                grad_ = cuda_shared_pointer<T>();
                grad_fn_ = nullptr;
            }
            TensorRaw(const cuda_shared_pointer<T> & data , const std::vector<size_t> & shape){
                shape_ = shape;
                size_ = get_strides_with_shape(shape);
                data_ = data;
                requires_grad_ = false;
                grad_ = cuda_shared_pointer<T>();
                grad_fn_ = nullptr;
            }
            TensorRaw(){
                shape_ = {};
                size_ = 0;
                strides = {};
                data_ = cuda_shared_pointer<T>();
                requires_grad_ = false;
                grad_ = cuda_shared_pointer<T>();
                grad_fn_ = nullptr;
            }
            int ref_count() const{
                return data_.ref_count();
            }
            template <typename U>
            friend TensorRaw<U> arange_raw(const U & start ,const U & end , const U & step , const Device device);
            template <typename U>
            friend TensorRaw<U> zeros_raw(const std::vector<size_t>& shape, const Device device );
            template <typename U>
            friend TensorRaw<U> ones_raw(const std::vector<size_t>& shape, const Device device );
            template <typename U>
            friend TensorRaw<U> rand_raw(const std::vector<size_t>& shape, const Device device );
            template <typename U>
            friend TensorRaw<U> randn_raw(const std::vector<size_t>& shape, const Device device );
            template <typename U>
            friend TensorRaw<U> full_raw(const std::vector<size_t>& shape, const U value ,  const Device device );
            // no need for destruction
            void print() const {
                auto original_flags = std::cout.flags();
                auto original_precision = std::cout.precision();
                std::cout << std::fixed << std::left << std::setprecision(4);
                int max_width = 12;
                TensorRaw<T> self = *this;
                if(self.device() == Cuda){
                    self = this->deepcopy();
                    self.to(Cpu);
                }
                if(self.shape_.size() == 0 && !self.data_.is_null()){
                    std::cout << std::setw(max_width) << self.data_[0] << "\n";
                    return;
                }
                else if(self.shape_.size() == 1){
                    std::cout  << "[\t";
                    for(int i = 0;i<self.shape_[0];i++){
                        std::cout << std::setw(max_width) << self.data_[i];
                        if(i != self.shape_[0] - 1) std::cout << ",";
                    }
                    std::cout << "\t]\n";
                }
                else if(self.shape_.size() ==  2){
                    std::cout << "[";
                    for(int i = 0;i<self.shape_[0];i++){
                        std::cout << "[\t";
                        for(int j = 0;j<self.shape_[1];j++){
                            std::cout << std::setw(max_width) << self.data_[i * self.shape_[1] + j];
                            if(j != self.shape_[1] - 1) std::cout << ",";
                        }
                        std::cout << "\t]";
                        if(i != self.shape_[0] - 1) std::cout << ",\n";
                    }
                    std::cout << "]\n";
                }
                else if(self.shape_.size() >= 3){
                    size_t step = strides[2];
                    size_t offset = 0;
                    std::cout << "[\n";
                    for(offset = 0;offset < self.size_;offset += step){
                        std::cout << "[";
                        for(int i = 0;i<self.shape_[self.shape_.size() - 2];i++){
                            std::cout << "[\t";
                            for(int j = 0;j<self.shape_[self.shape_.size() - 1];j++){
                                std::cout << std::setw(max_width) << self.data_[offset + i * self.shape_[self.shape_.size() - 1] + j];
                                if(j != self.shape_[self.shape_.size() - 1] - 1) std::cout << ",";
                            }
                            std::cout << "\t]";
                            if(i != self.shape_[self.shape_.size() - 2] - 1) std::cout << ",\n";
                        }
                        std::cout << "]";
                        if(offset != self.size_ - step) std::cout << ",\n";
                    }
                    std::cout << "]\n";
                }
                else{
                    throw std::runtime_error("print() on null tensor");
                }
                std::cout.flags(original_flags);
                std::cout.precision(original_precision);
            }
            TensorRaw<T> deepcopy() const{
                TensorRaw<T> result;
                result.data_ = data_.deepcopy();
                result.size_ = size_;
                result.strides = strides;
                result.shape_ =  shape_;
                result.requires_grad_ = requires_grad_;
                result.grad_ = grad_;
                result.grad_fn_ = grad_fn_;
                return result;
            }
    };
    template<typename T>
    TensorRaw<T> arange_raw(const T & start ,const T & end , const T & step = T(1) , const Device device = Cpu ){
        std::vector<T> values;
        for(T val = start;val < end;val += step){
            values.push_back(val);
        }
        std::vector<size_t> shape = { values.size() };
        TensorRaw<T> result(shape, device);
        if (device == Cpu){
            std::copy(values.begin(), values.end(), result.data_.get());
        }
        else{
            CHECK(cudaMemcpy(result.data_.get(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
        }
        return result;
    }
    template<typename T>
    TensorRaw<T> zeros_raw(const std::vector<size_t>& shape, const Device device = Cpu) {
        return TensorRaw<T>(shape, device);
    }
    template <typename T>
    __global__ void _fillWithOne(T* d_data, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            d_data[idx] = 1;
        }
    }

    template<typename T>
    TensorRaw<T> ones_raw(const std::vector<size_t>& shape, const Device device = Cpu) {
        TensorRaw<T> result(shape, device);
        if (device == Cpu){
            std::fill(result.data_.get(), result.data_.get() + result.size_, T(1));
        }
        else{
            _fillWithOne<T><<<CudaGetBlocks(result.size_), kCudaThreadsNum>>>(result.data_.get(), result.size_);
        }
        return result;
    }

    template<typename T>
    TensorRaw<T> rand_raw(const std::vector<size_t>& shape, const Device device = Cpu);
    template<>
    TensorRaw<float> rand_raw<float>(const std::vector<size_t>& shape, const Device device) {
        TensorRaw<float> result(shape, device);
        std::random_device rd;
        if (device == Cpu){
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(0.0, 1.0);
            for (size_t i = 0; i < result.size_; ++i) {
                result.data_[i] = dist(gen);
            }
        }
        else{
            curandGenerator_t rng;
            CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, (unsigned long long)rd()));
                CHECK_CURAND(curandGenerateUniform(rng, result.data_, result.size_));
            CHECK_CURAND(curandDestroyGenerator(rng));
        }
        return result;
    }
    template<>
    TensorRaw<double> rand_raw<double>(const std::vector<size_t>& shape, const Device device) {
        TensorRaw<double> result(shape, device);
        std::random_device rd;
        if (device == Cpu){
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < result.size_; ++i) {
                result.data_[i] = dist(gen);
            }
        }
        else{
            curandGenerator_t rng;
            CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, (unsigned long long)rd()));
            CHECK_CURAND(curandGenerateUniformDouble(rng, result.data_, result.size_));
            CHECK_CURAND(curandDestroyGenerator(rng));
        }
        return result;
    }

    template<typename T>
    TensorRaw<T> randn_raw(const std::vector<size_t>& shape, const Device device = Cpu) {
        TensorRaw<T> result(shape, device);
        if (device == Cpu){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(T(0), T(1));
            for (size_t i = 0; i < result.size_; ++i) {
                result.data_[i] = dist(gen);
            }
        }
        else{
            static_assert(std::is_same_v<T , float> || std::is_same_v<T , double> , "rand only support float and double");
            curandGenerator_t rng;
            CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, (unsigned long long)time(0)));
            if constexpr (std::is_same_v<T , float>){
                CHECK_CURAND(curandGenerateNormal(rng, result.data_.get(), result.size_ , T(0) , T(1)));
            }
            else {
                CHECK_CURAND(curandGenerateNormalDouble(rng, result.data_.get(), result.size_ , T(0) , T(1)));
            }
            CHECK_CURAND(curandDestroyGenerator(rng));
        }
        return result;
    }

    template <typename T>
    __global__ void _fillWithValue(T* d_data, int n, T value) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            d_data[idx] = value;
        }
    }
    template <typename U>
    TensorRaw<U> full_raw(const std::vector<size_t>& shape, const U value ,  const Device device ){
        TensorRaw<U> result(shape, device);
        if (device == Cpu){
            std::fill(result.get() , result.get() + result.size_ , value);
        }
        else{
            _fillWithValue<U><<<CudaGetBlocks(result.size_), kCudaThreadsNum>>>(result.get(), result.size_, value);
        }
        return result;
    }

    class MultiDimIndex{
        private:
            std::vector<size_t> index_;
            std::vector<size_t> shape_;
        public:
            MultiDimIndex(const std::vector<size_t> & index , const std::vector<size_t> & shape){
                index_ = index;
                shape_ = shape;
            }
            MultiDimIndex(const std::vector<size_t> & shape){
                shape_ = shape;
                index_ = std::vector<size_t>(shape_.size() , 0);
            }
            void next(){
                for(int i = index_.size() - 1;i>=0;i--){
                    if(index_[i] < shape_[i] - 1){
                        index_[i]++;
                        break;
                    }
                    else{
                        index_[i] = 0;
                    }
                }
            }
            std::vector<size_t> & get_index(){
                return index_;
            }
            const std::vector<size_t> & get_index() const{
                return index_;
            }
            bool operator==(const MultiDimIndex & other) const{
                if(shape_ != other.shape_){
                    return false;
                }
                return index_ == other.index_;
            }
            bool operator!=(const MultiDimIndex & other) const{
                return !(*this == other);
            }
            bool is_zero() const{
                for(int i = 0;i<index_.size();i++){
                    if(index_[i] != 0){
                        return false;
                    }
                }
                return true;
            }
            size_t calculate_offset(const std::vector<size_t> & strides = {}) const{
                if(strides.empty()){
                    auto newstrides = std::vector<size_t>();
                    size_t nowstride = 1;
                    for(int i = shape_.size() - 1;i>=0;i--){
                        newstrides.push_back(nowstride);
                        nowstride *= shape_[i];
                    }
                    size_t offset = 0;
                    for(int i = 0;i<index_.size();i++){
                        offset += index_[i] * strides[shape_.size() - 1 - i];
                    }
                    return offset;
                }
                size_t offset = 0;
                for(int i = 0;i<index_.size();i++){
                    offset += index_[i] * strides[shape_.size() - 1 - i];
                }
                return offset;
            }
    };
    template <typename T>
    class Tensor{
        private:
            std::shared_ptr<TensorRaw<T>> data_ptr_;
        public:
            Tensor(){
                data_ptr_ = nullptr;
            }
            friend class nn::Functional::AddFunc<T>;
            friend class nn::Functional::Function<T>;
            friend class nn::Functional::NegFunc<T>;
            friend class nn::Functional::SubFunc<T>;
            friend class nn::Functional::MulFunc<T>;
            friend class nn::Functional::DivFunc<T>;
            friend class nn::Functional::ReLUFunc<T>;
            friend class nn::Functional::SigmoidFunc<T>;
            friend class nn::Functional::TransposeFunc<T>;
            friend class nn::Functional::ReshapeFunc<T>;
            friend class nn::Functional::MatmulFunc<T>;
            size_t size() const{
                if(!data_ptr_){
                    throw std::runtime_error("size() on null tensor");
                }
                return data_ptr_->size();
            }



            Tensor(const Tensor & other) {
                data_ptr_ = other.data_ptr_;
            }
            Tensor & operator=(const Tensor& other) {
                if(this == &other) return *this;
                data_ptr_ = other.data_ptr_;
                return *this;
            }
            Device device() const{
                if(!data_ptr_){
                    throw std::runtime_error("device() on null tensor");
                }
                return data_ptr_->device();
            }
            const std::vector<size_t> & shape() const{
                if(!data_ptr_){
                    throw std::runtime_error("shape() on null tensor");
                }
                return data_ptr_->shape();
            }
            void to(Device device){
                if(!data_ptr_){
                    throw std::runtime_error("to() on null tensor");
                }
                data_ptr_->to(device);
            }

            T * get(){
                if(!data_ptr_){
                    throw std::runtime_error("get() on null tensor");
                }
                return data_ptr_->get();
            }
            const T * get() const{
                if(!data_ptr_){
                    throw std::runtime_error("get() on null tensor");
                }
                return data_ptr_->get();
            }
            std::vector<size_t> get_strides() const{
                if(!data_ptr_){
                    throw std::runtime_error("get_strides() on null tensor");
                }
                return data_ptr_->get_strides();
            }


            Tensor(const std::vector<size_t> & shape ,const Device device=Cpu){
                data_ptr_ = std::make_shared<TensorRaw<T>>(shape , device);
            }
            int ref_count() const{
                return data_ptr_.use_count();
            }
            template <typename U>
            friend Tensor<U> arange(const U & start ,const U & end , const U & step , const Device device);
            template <typename U>
            friend Tensor<U> zeros(const std::vector<size_t>& shape, const Device device );
            template <typename U>
            friend Tensor<U> ones(const std::vector<size_t>& shape, const Device device );
            template <typename U>
            friend Tensor<U> rand(const std::vector<size_t>& shape, const Device device );
            template <typename U>
            friend Tensor<U> randn(const std::vector<size_t>& shape, const Device device );
            template <typename U>
            friend Tensor<U> full(const std::vector<size_t>& shape, const U value  , const Device device );
            // no need for destruction
            void print() const {
                if(!data_ptr_){
                    throw std::runtime_error("print() on null tensor");
                }
                data_ptr_->print();
            }
            Tensor<T> deepcopy() const{
                Tensor<T> result;
                result.data_ptr_ = std::make_shared<TensorRaw<T>>(data_ptr_->deepcopy());
                return result;
            }

            Tensor<T> operator+(const Tensor<T> & b) const;
            Tensor<T> operator-() const;
            Tensor<T> operator-(const Tensor<T> & b) const;
            Tensor<T> operator*(const Tensor<T> & b) const;
            Tensor<T> operator/(const Tensor<T> & b) const;
            Tensor<T> relu() const;
            Tensor<T> sigmoid() const;
            Tensor<T> transpose(const std::vector<size_t> & perm = {}) const;
            Tensor<T> reshape(const std::vector<size_t> & newshape) const;
            Tensor<T> matmul(const Tensor<T> & b) const;
            Tensor<T> pool2d(const std::vector<size_t> & kernel_shape) const;
            Tensor<T> expand(const size_t axis) const{
                if(!data_ptr_){
                    throw std::runtime_error("expand() on null tensor");
                }
                auto oldshape = data_ptr_->shape();
                if(axis > oldshape.size()){
                    throw std::runtime_error("expand() axis out of range");
                }
                std::vector<size_t> newshape = oldshape;
                newshape.insert(newshape.begin() + axis , 1);
                return this->reshape(newshape);
            }
            bool requires_grad() const{
                if(!data_ptr_){
                    throw std::runtime_error("requires_grad() on null tensor");
                }
                return data_ptr_->requires_grad();
            }
            void set_requires_grad(bool requires_grad){
                if(!data_ptr_){
                    throw std::runtime_error("set_requires_grad() on null tensor");
                }
                data_ptr_->set_requires_grad(requires_grad);
            }
            void set_grad_fn(const std::shared_ptr<nn::Functional::Function<T>> & grad_fn){
                if(!data_ptr_){
                    throw std::runtime_error("set_grad_fn() on null tensor");
                }
                data_ptr_->set_grad_fn(grad_fn);
            }
            std::shared_ptr<nn::Functional::Function<T>> get_grad_fn() const{
                if(!data_ptr_){
                    throw std::runtime_error("get_grad_fn() on null tensor");
                }
                return data_ptr_->get_grad_fn();
            }
            cuda_shared_pointer<T> & get_shared_ptr(){
                if(!data_ptr_){
                    throw std::runtime_error("get_shared_ptr() on null tensor");
                }
                return data_ptr_->get_shared_ptr();
            }
            const cuda_shared_pointer<T> & get_shared_ptr() const{
                if(!data_ptr_){
                    throw std::runtime_error("get_shared_ptr() on null tensor");
                }
                return data_ptr_->get_shared_ptr();
            }
            void set_grad(const cuda_shared_pointer<T> & grad){
                if(!data_ptr_){
                    throw std::runtime_error("set_grad() on null tensor");
                }
                data_ptr_->set_grad(grad);
            }
            void set_grad(const Tensor<T> & grad){
                if(!data_ptr_){
                    throw std::runtime_error("set_grad() on null tensor");
                }
                data_ptr_->set_grad(grad.get_shared_ptr());
            }
            const cuda_shared_pointer<T> & get_grad() const{
                if(!data_ptr_){
                    throw std::runtime_error("get_grad() on null tensor");
                }
                return data_ptr_->get_grad();
            }
            Tensor<T> get_grad_tensor() const{
                if(!data_ptr_){
                    throw std::runtime_error("get_grad_tensor() on null tensor");
                }
                Tensor<T> result;
                result.data_ptr_ = std::make_shared<TensorRaw<T>>(data_ptr_->get_grad() , data_ptr_->shape());
                return result;
            }

            size_t ndim() const{
                if(!data_ptr_){
                    throw std::runtime_error("ndim() on null tensor");
                }
                return data_ptr_->shape().size();
            }
            Tensor<T>(const T * data , const std::vector<size_t> & shape , const Device device = Cpu){
                data_ptr_ = std::make_shared<TensorRaw<T>>(data , shape , device);
            }

            Tensor<T>(T value , const std::vector<size_t> & shape = {1} , const Device device = Cpu){
                data_ptr_ = std::make_shared<TensorRaw<T>>(full_raw<T>(shape , value , device));
            }
            Tensor<T> & operator=(const T value){
                data_ptr_ = std::make_shared<TensorRaw<T>>(full_raw<T>({1} , value , Cpu));
                return *this;
            }
    };

    template <typename U>
    Tensor<U> arange(const U & start ,const U & end , const U & step , const Device device = Cpu ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(arange_raw<U>(start , end , step , device));
        return result;
    }
    template <typename U>
    Tensor<U> zeros(const std::vector<size_t>& shape, const Device device = Cpu ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(zeros_raw<U>(shape , device));
        return result;
    }
    template <typename U>
    Tensor<U> ones(const std::vector<size_t>& shape, const Device device = Cpu){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(ones_raw<U>(shape , device));
        return result;
    }
    template <typename U>
    Tensor<U> rand(const std::vector<size_t>& shape, const Device device = Cpu ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(rand_raw<U>(shape , device));
        return result;
    }
    template <typename U>
    Tensor<U> randn(const std::vector<size_t>& shape, const Device device = Cpu ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(randn_raw<U>(shape , device));
        return result;
    }
    template <typename U>
    Tensor<U> full(const std::vector<size_t>& shape, const U value = U(1) , const Device device = Cpu ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(full_raw<U>(shape , value , device));
        return result;
    }


}

#endif