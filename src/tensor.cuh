#ifndef _TENSOR_H_
#define _TENSOR_H_

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
            template<typename T>
            class SumFunc;
        }
    };
    

    __host__ __device__ __forceinline__
    int clz32_nonbuiltin(uint32_t x) {
        if (x == 0) return 32;
        int n = 0;
        if ((x & 0xFFFF0000u) == 0) { n += 16; x <<= 16; }
        if ((x & 0xFF000000u) == 0) { n += 8;  x <<= 8;  }
        if ((x & 0xF0000000u) == 0) { n += 4;  x <<= 4;  }
        if ((x & 0xC0000000u) == 0) { n += 2;  x <<= 2;  }
        if ((x & 0x80000000u) == 0) { n += 1;               }
        return n;
    }

    __host__ __device__ __forceinline__
    int clz64_nonbuiltin(uint64_t x) {
        if (x == 0) return 64;
        int n = 0;
        uint32_t hi = (uint32_t)(x >> 32);
        if (hi == 0) {
            n += 32;
            uint32_t lo = (uint32_t)x;
            if ((lo & 0xFFFF0000u) == 0) { n += 16; lo <<= 16; }
            if ((lo & 0xFF000000u) == 0) { n += 8;  lo <<= 8;  }
            if ((lo & 0xF0000000u) == 0) { n += 4;  lo <<= 4;  }
            if ((lo & 0xC0000000u) == 0) { n += 2;  lo <<= 2;  }
            if ((lo & 0x80000000u) == 0) { n += 1;               }
            return n;
        } else {
            if ((hi & 0xFFFF0000u) == 0) { n += 16; hi <<= 16; }
            if ((hi & 0xFF000000u) == 0) { n += 8;  hi <<= 8;  }
            if ((hi & 0xF0000000u) == 0) { n += 4;  hi <<= 4;  }
            if ((hi & 0xC0000000u) == 0) { n += 2;  hi <<= 2;  }
            if ((hi & 0x80000000u) == 0) { n += 1;               }
            return n;
        }
    }

    const size_t kCudaThreadsNum = 1024;
    __host__ __device__ inline int CudaGetBlocks(const int N) {
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
            fprintf(stderr, "cuRAND Error: %d (line: %d) in file %s\n", err, __LINE__ , __FILE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
    #define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error: %d (line: %d) in file %s\n", err , __LINE__ , __FILE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

    template <typename T>
    __global__ void _fillWithValue(T* d_data, int n, T value) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            d_data[idx] = value;
        }
    }

    enum Device{
        Cpu , Cuda
    };

    extern Device DefaultDevice;
    static int tensor_count = 0;

    /* cuda_shared_pointer for memory management */
    template <typename T>
    struct less_addr;

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
                }
            }
            void release(){
                if (ref_count_ && (--(*ref_count_) == 0)){
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
            cuda_shared_pointer(const size_t size ,const Device device = DefaultDevice , const bool need_init = true){
                allocate(size , device);
                if(need_init){
                    if(device == Cpu){
                        memset(data_ , 0 , size_ * sizeof(T));
                    }
                    else{
                        CHECK(cudaMemset(data_ , 0 , size_ * sizeof(T)));
                    }
                }
            }
            cuda_shared_pointer(const T * data ,const size_t size ,const Device device = DefaultDevice , const Device datadevice = Cpu){
                allocate(size , device);
                size_ = size;
                device_ = device;
                ref_count_ = std::make_shared<int>(1);
                if(datadevice == Cpu){
                    if(device == Cpu){
                        memcpy(data_ , data , size_ * sizeof(T));
                    }
                    else{
                        CHECK(cudaMemcpy(data_ , data , size_ * sizeof(T) , cudaMemcpyHostToDevice));
                    }
                }
                else{
                    if(device == Cpu){
                        CHECK(cudaMemcpy(data_ , data , size_ * sizeof(T) , cudaMemcpyDeviceToHost));
                    }
                    else{
                        CHECK(cudaMemcpy(data_ , data , size_ * sizeof(T) , cudaMemcpyDeviceToDevice));
                    }
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
                if(ref_count_){
                    (*ref_count_)++;
                }
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
            cuda_shared_pointer(const std::vector<T> & data , const Device device = DefaultDevice){
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
            void inplace_zero(){
                if(!data_){
                    std::cerr << "cuda_shared_pointer is null! Cannot inplace_zero!" << std::endl;
                    throw std::runtime_error("cuda_shared_pointer is null! Cannot inplace_zero!");
                }
                if(device_ == Cpu){
                    memset(data_ , 0 , size_ * sizeof(T));
                }
                else{
                    CHECK(cudaMemset(data_ , 0 , size_ * sizeof(T)));
                }
            }
    };




    template <typename T>
    class TensorRaw{
        private:
            size_t size_;
            std::vector<int> shape_;
            std::vector<int> strides;
            cuda_shared_pointer<T> data_;
            bool requires_grad_;
            cuda_shared_pointer<T> grad_;
            std::shared_ptr<nn::Functional::Function<T>> grad_fn_;
            int get_strides_with_shape(const std::vector<int> & shape){
                strides = {};
                int nowstride = 1;
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



            void inplace_zero(){
                data_.inplace_zero();
            }

            TensorRaw<T>(const T * data , const std::vector<int> & shape , const Device device = DefaultDevice){
                size_ = get_strides_with_shape(shape);
                shape_ = shape;
                data_ = cuda_shared_pointer<T>(data , size_ , device);
                requires_grad_ = false;
            }
            TensorRaw<T>(const T value , const std::vector<int> & shape = {1} , const Device device = DefaultDevice){
                size_ = get_strides_with_shape(shape);
                shape_ = shape;
                data_ = cuda_shared_pointer<T>(size_ , device , false);
                _fillWithValue<<<CudaGetBlocks(size_), CudaGetThreads()>>>(data_.get(), size_, value);
                requires_grad_ = false;
            }
            TensorRaw<T>(cuda_shared_pointer<T> & data , const std::vector<int> & shape){// create a view 
                size_ = get_strides_with_shape(shape);
                if(data.is_null() || data.size() != size_){
                    std::cerr << "Invalid view!" << std::endl;
                    throw std::runtime_error("Invalid view!");
                }
                shape_ = shape;
                data_ = data;
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
            void set_grad(const TensorRaw<T> & grad){
                grad_ = grad.data_;
            }
            const cuda_shared_pointer<T> & get_grad() const{
                return grad_;
            }
            cuda_shared_pointer<T> & get_grad(){
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
            const std::vector<int> & shape() const{
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
            std::vector<int> get_strides() const{
                return strides;
            }

            
            TensorRaw(const std::vector<int> & shape ,const Device device=DefaultDevice){
                shape_ = shape;
                size_ = get_strides_with_shape(shape);
                data_ = cuda_shared_pointer<T>(size_ , device);
                requires_grad_ = false;
                grad_ = cuda_shared_pointer<T>();
                grad_fn_ = nullptr;
            }
            TensorRaw(const std::vector<int> & shape , const bool need_init ,const Device device){
                shape_ = shape;
                size_ = get_strides_with_shape(shape);
                data_ = cuda_shared_pointer<T>(size_ , device , need_init);
                requires_grad_ = false;
                grad_ = cuda_shared_pointer<T>();
                grad_fn_ = nullptr;
            }
            TensorRaw(const cuda_shared_pointer<T> & data , const std::vector<int> & shape){
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
            friend TensorRaw<U> zeros_raw(const std::vector<int>& shape, const Device device );
            template <typename U>
            friend TensorRaw<U> ones_raw(const std::vector<int>& shape, const Device device );
            template <typename U>
            friend TensorRaw<U> rand_raw(const std::vector<int>& shape, const Device device );
            template <typename U>
            friend TensorRaw<U> randn_raw(const std::vector<int>& shape, const Device device);
            template <typename U>
            friend TensorRaw<U> full_raw(const std::vector<int>& shape, const U value ,  const Device device );
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
                if (self.shape_.size() == 0 && !self.data_.is_null()) {
                    std::cout << "tensor(" << self.data_[0] << ")\n";
                    return;
                }
                else if (self.shape_.size() == 1) {
                    std::cout << "tensor([";
                    for (int i = 0; i < self.shape_[0]; i++) {
                        std::cout << std::setw(max_width) << self.data_[i];
                        if (i != self.shape_[0] - 1) std::cout << ", ";
                    }
                    std::cout << "])\n";
                }
                else if (self.shape_.size() == 2) {
                    std::cout << "tensor([\n";
                    for (int i = 0; i < self.shape_[0]; i++) {
                        std::cout << "  [";
                        for (int j = 0; j < self.shape_[1]; j++) {
                            std::cout << std::setw(max_width) << self.data_[i * self.shape_[1] + j];
                            if (j != self.shape_[1] - 1) std::cout << ", ";
                        }
                        std::cout << "]";
                        if (i != self.shape_[0] - 1) std::cout << ",\n";
                    }
                    std::cout << "\n])\n";
                }
                else if (self.shape_.size() >= 3) {
                    std::cout << "tensor([\n";
                     int inner = self.shape_[self.shape_.size() - 1];
                    int mid   = self.shape_[self.shape_.size() - 2];
                    int step  = inner * mid;

                    for (int b = 0; b < self.size_ / step; b++) {
                        int offset = b * step;

                        std::cout << "  [\n";
                        for (int i = 0; i < mid; i++) {
                            std::cout << "    [";
                            for (int j = 0; j < inner; j++) {
                                std::cout << std::setw(max_width)
                                        << self.data_[offset + i * inner + j];
                                if (j != inner - 1) std::cout << ", ";
                            }
                            std::cout << "]";
                            if (i != mid - 1) std::cout << ",\n";
                        }
                        std::cout << "\n  ]";
                        if (b != self.size_ / step - 1) std::cout << ",\n\n";
                    }
                    std::cout << "\n])\n";
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

            void zero_grad(cudaStream_t stream = 0){
                if(grad_.is_null()){
                    grad_ = cuda_shared_pointer<T>(this->size() , this->device());
                }
                else{
                    if(this->device() == Cuda){
                        CHECK(cudaMemsetAsync(grad_.get() , 0 , sizeof(T) * this->size() , stream));
                    }
                    else{
                        std::fill(grad_.get() , grad_.get() + this->size() , T(0));
                    }
                }
            }
            void alloc_grad(){
                if(grad_.is_null()){
                    grad_ = cuda_shared_pointer<T>(this->size() , this->device() , false);
                }
            }
    };


    template <typename T>
    __global__ void __arange_kernel(T * output , const T start , const T step , const int n){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(idx < n){
            output[idx] = start + idx * step;
        }
    }
    template<typename T>
    TensorRaw<T> arange_raw(const T & start ,const T & end , const T & step = T(1) , const Device device = DefaultDevice ){
        float nf = (end - start) / step;
        int n = static_cast<int>(std::ceil(nf));
        TensorRaw<T> result({n}, device);
        if (device == Cpu){
            int idx = 0;
            for(float i = start ;i < end;i += step){
                result.get()[idx++] = i;
            }
        }
        else{
            __arange_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(result.get(), start, step, n);
        }
        return result;
    }
    template<typename T>
    TensorRaw<T> zeros_raw(const std::vector<int>& shape, const Device device = DefaultDevice) {
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
    TensorRaw<T> ones_raw(const std::vector<int>& shape, const Device device = DefaultDevice) {
        TensorRaw<T> result(shape, false, device);
        if (device == Cpu){
            std::fill(result.data_.get(), result.data_.get() + result.size_, T(1));
        }
        else{
            _fillWithOne<T><<<CudaGetBlocks(result.size_), kCudaThreadsNum>>>(result.data_.get(), result.size_);
        }
        return result;
    }
    template<typename T>
    TensorRaw<T> rand_raw(const std::vector<int>& shape, const Device device){
        return TensorRaw<T>(shape, device);
    }

    template<>
    inline TensorRaw<float> rand_raw<float>(const std::vector<int>& shape, const Device device) {
        TensorRaw<float> result(shape,0, device);
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
    inline TensorRaw<double> rand_raw<double>(const std::vector<int>& shape, const Device device) {
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
    TensorRaw<T> randn_raw(const std::vector<int>& shape, const Device device) {
        TensorRaw<T> result(shape, false ,  device);
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
            std::random_device rd;
            CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, (unsigned long long)rd()));
            size_t roundSize = 1 << (64 - clz64_nonbuiltin(result.size_) );
            if(roundSize == 2 * result.size_ && result.size_ != 1){
                roundSize = result.size_;
            }
            if constexpr (std::is_same_v<T , float>){
                T * data;
                CHECK(cudaMalloc(&data , roundSize * sizeof(T)));
                CHECK_CURAND(curandGenerateNormal(rng, data, roundSize , T(0) , T(1)));
                CHECK(cudaMemcpy(result.get() , data , result.size_ * sizeof(T) , cudaMemcpyDeviceToDevice));
                CHECK(cudaFree(data));
            }
            else {
                T * data;
                CHECK(cudaMalloc(&data , roundSize * sizeof(T)));
                CHECK_CURAND(curandGenerateNormalDouble(rng, data, roundSize , T(0) , T(1)));
                CHECK(cudaMemcpy(result.get() , data , result.size_ * sizeof(T) , cudaMemcpyDeviceToDevice));
                CHECK(cudaFree(data));
            }
            CHECK_CURAND(curandDestroyGenerator(rng));
        }
        return result;
    }

    template <typename U>
    TensorRaw<U> full_raw(const std::vector<int>& shape, const U value ,  const Device device ){
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
            std::vector<int> index_;
            std::vector<int> shape_;
        public:
            MultiDimIndex(const std::vector<int> & index , const std::vector<int> & shape){
                index_ = index;
                shape_ = shape;
            }
            MultiDimIndex(const std::vector<int> & shape){
                shape_ = shape;
                index_ = std::vector<int>(shape_.size() , 0);
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
            std::vector<int> & get_index(){
                return index_;
            }
            const std::vector<int> & get_index() const{
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
             int calculate_offset(const std::vector<int> & strides = {}) const{
                if(strides.empty()){
                    auto newstrides = std::vector<int>();
                    int nowstride = 1;
                    for(int i = shape_.size() - 1;i>=0;i--){
                        newstrides.push_back(nowstride);
                        nowstride *= shape_[i];
                    }
                    int offset = 0;
                    for(int i = 0;i<index_.size();i++){
                        offset += index_[i] * strides[shape_.size() - 1 - i];
                    }
                    return offset;
                }
                 int offset = 0;
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
            friend class nn::Functional::SumFunc<T>;
            friend struct less_addr<T>;
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
            const std::vector<int> & shape() const{
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
            std::vector<int> get_strides() const{
                if(!data_ptr_){
                    throw std::runtime_error("get_strides() on null tensor");
                }
                return data_ptr_->get_strides();
            }

            void inplace_zero(){
                if(!data_ptr_){
                    throw std::runtime_error("inplace_zero() on null tensor");
                }
                data_ptr_->inplace_zero();
            }


            Tensor(const std::vector<int> & shape ,const Device device=DefaultDevice){
                data_ptr_ = std::make_shared<TensorRaw<T>>(shape , device);
            }
            Tensor(const std::vector<int> & shape , const bool need_init , const Device device){
                data_ptr_ = std::make_shared<TensorRaw<T>>(shape , need_init , device);
            }
            int ref_count() const{
                return data_ptr_.use_count();
            }
            template <typename U>
            friend Tensor<U> arange(const U & start ,const U & end , const U & step , const Device device);
            template <typename U>
            friend Tensor<U> zeros(const std::vector<int>& shape, const Device device );
            template <typename U>
            friend Tensor<U> ones(const std::vector<int>& shape, const Device device );
            template <typename U>
            friend Tensor<U> rand(const std::vector<int>& shape, const Device device );
            template <typename U>
            friend Tensor<U> randn(const std::vector<int>& shape, const Device device );
            template <typename U>
            friend Tensor<U> full(const std::vector<int>& shape, const U value  , const Device device );
            template <typename U>
            friend Tensor<U> make_view(const cuda_shared_pointer<U> & data , const std::vector<int> & shape);
            // no need for destruction
            void print() const {
                if(!data_ptr_){
                    throw std::runtime_error("print() on null tensor");
                }
                data_ptr_->print();
            }
            Tensor<T> deepcopy() const{
                if(!data_ptr_){
                    return Tensor<T>();
                }
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
            Tensor<T> transpose(const std::vector<int> & perm = {}) const;
            Tensor<T> reshape(const std::vector<int> & newshape) const;
            Tensor<T> matmul(const Tensor<T> & b) const;
            Tensor<T> maxpool2d(const int kernel_height , const int kernel_width , const int pad_h = 0 , const int pad_w = 0 , const int stride_h = 0 , const int stride_w = 0) const;
            Tensor<T> avgpool2d(const int kernel_height , const int kernel_width , const int pad_h = 0 , const int pad_w = 0 , const int stride_h = 0 , const int stride_w = 0) const;
            Tensor<T> sum(const int  axis) const;
            Tensor<T> expand(const int axis) const{
                if(!data_ptr_){
                    throw std::runtime_error("expand() on null tensor");
                }
                auto oldshape = data_ptr_->shape();
                if(axis > oldshape.size()){
                    throw std::runtime_error("expand() axis out of range");
                }
                std::vector<int> newshape = oldshape;
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
            void set_grad_(const cuda_shared_pointer<T> & grad){
                if(!data_ptr_){
                    throw std::runtime_error("set_grad_() on null tensor");
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
            cuda_shared_pointer<T> & get_grad(){
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

             int ndim() const{
                if(!data_ptr_){
                    throw std::runtime_error("ndim() on null tensor");
                }
                return data_ptr_->shape().size();
            }
            Tensor<T>(const T * data , const std::vector<int> & shape , const Device device = DefaultDevice){
                data_ptr_ = std::make_shared<TensorRaw<T>>(data , shape , device);
            }

            Tensor<T>(T value , const std::vector<int> & shape = {1} , const Device device = DefaultDevice){
                data_ptr_ = std::make_shared<TensorRaw<T>>(full_raw<T>(shape , value , device));
            }
            Tensor<T> & operator=(const T value){
                data_ptr_ = std::make_shared<TensorRaw<T>>(full_raw<T>({1} , value , DefaultDevice));
                return *this;
            }
            T & operator[](const int index){
                if(!data_ptr_){
                    throw std::runtime_error("operator[] on null tensor");
                }
                return data_ptr_->get()[index];
            }
            const T & operator[](const int index) const{
                if(!data_ptr_){
                    throw std::runtime_error("operator[] on null tensor");
                }
                return data_ptr_->get()[index];
            }
            bool is_null() const{
                return data_ptr_ == nullptr;
            }
            void backward(const Tensor<T> & grad_out = Tensor<T>());
            void zero_grad(cudaStream_t stream = 0){
                if(!data_ptr_){
                    throw std::runtime_error("zero_grad() on null tensor");
                }
                data_ptr_->zero_grad(stream);

            }
            void alloc_grad(){
                if(!data_ptr_){
                    throw std::runtime_error("alloc_grad() on null tensor");
                }
                data_ptr_->alloc_grad();
            }
            friend Tensor<T> operator+(const T & a, const Tensor<T> & b){
                Tensor<T> a_tensor(a , b.shape() , b.device());
                return a_tensor + b;
            }
            friend Tensor<T> operator+(const Tensor<T> & a, const T & b){
                Tensor<T> b_tensor(b , a.shape() , a.device());
                return a + b_tensor;
            }
            friend Tensor<T> operator-(const T & a, const Tensor<T> & b){
                Tensor<T> a_tensor(a , b.shape() , b.device());
                return a_tensor - b;
            }
            friend Tensor<T> operator-(const Tensor<T> & a, const T & b){
                Tensor<T> b_tensor(b , a.shape() , a.device());
                return a - b_tensor;
            }
            friend Tensor<T> operator*(const T & a, const Tensor<T> & b){
                Tensor<T> a_tensor(a , b.shape() , b.device());
                return a_tensor * b;
            }
            friend Tensor<T> operator*(const Tensor<T> & a, const T & b){
                Tensor<T> b_tensor(b , a.shape() , a.device());
                return a * b_tensor;
            }
            friend Tensor<T> operator/(const T & a, const Tensor<T> & b){
                Tensor<T> a_tensor(a , b.shape() , b.device());
                return a_tensor / b;
            }
            friend Tensor<T> operator/(const Tensor<T> & a, const T & b){
                Tensor<T> b_tensor(b , a.shape() , a.device());
                return a / b_tensor;
            }
            T item() const{
                if(!data_ptr_){
                    std::cerr << "item() on null tensor" << std::endl;
                    throw std::runtime_error("item() on null tensor");
                }
                if(data_ptr_->device() == Cpu){
                    return data_ptr_->get()[0];
                }
                else{
                    T result;
                    CHECK(cudaMemcpy(&result , data_ptr_->get() , sizeof(T) , cudaMemcpyDeviceToHost));
                    return result;
                }
            }

            size_t __get_shared_id() const{
                return (size_t)data_ptr_.get();
            }
    };

    template <typename U>
    Tensor<U> arange(const U & start ,const U & end , const U & step , const Device device = DefaultDevice ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(arange_raw<U>(start , end , step , device));
        return result;
    }
    template <typename U>
    Tensor<U> zeros(const std::vector<int>& shape, const Device device = DefaultDevice ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(zeros_raw<U>(shape , device));
        return result;
    }
    template <typename U>
    Tensor<U> ones(const std::vector<int>& shape, const Device device = DefaultDevice){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(ones_raw<U>(shape , device));
        return result;
    }
    template <typename U>
    Tensor<U> rand(const std::vector<int>& shape, const Device device = DefaultDevice ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(rand_raw<U>(shape , device));
        return result;
    }
    template <typename U>
    Tensor<U> randn(const std::vector<int>& shape, const Device device = DefaultDevice ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(randn_raw<U>(shape , device));
        return result;
    }
    template <typename U>
    Tensor<U> full(const std::vector<int>& shape, const U value = U(1) , const Device device = DefaultDevice ){
        Tensor<U> result;
        result.data_ptr_ = std::make_shared<TensorRaw<U>>(full_raw<U>(shape , value , device));
        return result;
    }

    template <typename T>
    struct less_addr{
        bool operator()(const Tensor<T> & a, const Tensor<T> & b) const{
            return a.data_ptr_.get() < b.data_ptr_.get();
        }
    };
    template <typename T>
    Tensor<T> make_view(const cuda_shared_pointer<T> & data , const std::vector<int> & shape){
        Tensor<T> result;
        result.data_ptr_ = std::make_shared<TensorRaw<T>>(data , shape);
        return result;
    }

}

#endif