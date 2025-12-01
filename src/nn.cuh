#ifndef _NN_H_
#define _NN_H_

#include "tensor.cuh"
#include <cublas_v2.h>
#include <cmath>

namespace mytorch{
namespace nn{
    constexpr int kCudaTransposeTileSize = 4;
    constexpr int kCudaMultiDimMax = 16;
    constexpr int kStreamCount = 8;

    // defines general math functions for use
    template <typename T>
    __device__ __host__ inline T nn_rsqrt(T x){
        return rsqrt(x);
    }
    template <> 
    __device__ __host__ inline float nn_rsqrt(float x){
        return rsqrtf(x);
    }

    template <typename T>
    __device__ __host__ inline T nn_sqrt(T x){
        return sqrt(x);
    }
    template <> 
    __device__ __host__ inline float nn_sqrt(float x){
        return sqrtf(x);
    }

    template <typename T>
    __device__ __host__ inline T nn_exp(T x){
        return exp(x);
    }
    template <> 
    __device__ __host__ inline float nn_exp(float x){
        return expf(x);
    }
    template <typename T>
    __device__ inline T nn_exp_device(T x){
        return exp(x);
    }
    template <>
    __device__ inline float nn_exp_device(float x){
        return __expf(x);
    }


    // initialization

    template __device__ __host__ float nn_rsqrt<float>(float x);
    template __device__ __host__ double nn_rsqrt<double>(double x);
    template __device__ __host__ float nn_sqrt<float>(float x);
    template __device__ __host__ double nn_sqrt<double>(double x);
    template __device__ __host__ float nn_exp<float>(float x);
    template __device__ __host__ double nn_exp<double>(double x);


    template <typename T>
    class Module{
        protected:
            bool training;
        public:
            Module(){ training = true; };
            virtual Tensor<T> forward(const std::vector<Tensor<T>> & input){
                return Tensor<T>();
            }
            Tensor<T> operator()(const std::vector<Tensor<T>> & input){
                return forward(input);
            }
            Tensor<T> operator()(const Tensor<T> & input){
                return forward({input});
            }
            virtual std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out){
                return {}; // default no backward
            }
            virtual std::vector<Tensor<T>> parameters(){return {};};
            virtual void train(){
                training = true;
            }
            virtual void eval(){
                training = false;
            }
            virtual void zero_grad(){
                for(auto & param : parameters()){
                    param.zero_grad();
                }
            }
    };

    template <typename T >
    __device__ T warpReduceMax(T val);
    template <typename T >
    __device__ T warpReduceSum(T val);

    template <typename T>
    class Sequential : public Module<T>{
        private:
            std::vector<std::shared_ptr<Module<T>>> modules_;
            std::vector<Tensor<T>> params;
        public:
            Sequential(const std::vector<std::shared_ptr<Module<T>>> & modules){
                modules_ = modules;
                for(auto & module : modules_){
                    auto module_params = module->parameters();
                    params.insert(params.end() , module_params.begin() , module_params.end());
                }
            }
            Sequential(){}
            Tensor<T> forward(const std::vector<Tensor<T>> & input){
                Tensor<T> output = input[0];
                for(auto & module : modules_){
                    output = module->forward({output});
                }
                return output;
            }
            std::vector<Tensor<T>> parameters() override{
                return params;
            }
            void train() override{
                training = true;
                for(auto & module : modules_){
                    module->train();
                }
            }
            void eval() override{
                training = false;
                for(auto & module : modules_){
                    module->eval();
                }
            }
    };

    
    class CudaMultiDimIndex{
        private:
            int ndim_;
            int index_[kCudaMultiDimMax];
            int shape_[kCudaMultiDimMax];
        public:
            __device__ CudaMultiDimIndex(const int * shape ,const int ndim){
                ndim_ = ndim;
                for(int i = 0;i<ndim_;i++){
                    shape_[i] = shape[i];
                    index_[i] = 0;
                }
            }
            __device__ int * get_index(){
                return index_;
            }
            __device__ void next(){
                for(int i = ndim_ - 1;i>=0;i--){
                    if(index_[i] < shape_[i] - 1){
                        index_[i]++;
                        break;
                    }
                    else{
                        index_[i] = 0;
                    }
                }
            }
            __device__ bool is_zero() const{
                for(int i = 0;i<ndim_;i++){
                    if(index_[i] != 0){
                        return false;
                    }
                }
                return true;
            }
            __device__ int calculate_offset(const int * strides) const{
                int offset = 0;
                for(int i = 0;i<ndim_;i++){
                    offset += index_[i] * strides[ndim_ - 1 - i];
                }
                return offset;
            }
            __device__ int operator[](int i) const{
                return index_[i];
            }
    };


    namespace Functional{


        template <typename T>
        __global__ void _neg_forward_kernel(T * output , const T * input , const int size){
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if(index < size){
                output[index] = -input[index];
            }
        }
        template <typename T>
        class Function{
            public:
                virtual ~Function() = default;
                virtual Tensor<T> forward(const std::vector<Tensor<T>> & inputs){
                    return Tensor<T>();
                };
                virtual std::vector<Tensor<T>> backward(const Tensor<T>& grad_output){
                    return {};
                }
                virtual std::vector<Tensor<T>> get_inputs() const{
                    return {};
                }
                virtual Tensor<T> operator()(const std::vector<Tensor<T>> & inputs){
                    return forward(inputs);
                }
                virtual Tensor<T> operator()(const Tensor<T> & input){
                    return forward({input});
                }
        };


        template <typename T>
        class ModuleFunctionWrapper : public Function<T>{
            private:
                Module<T> * module_;
                Tensor<T> input_;
            public:
                ModuleFunctionWrapper(Module<T> * m , const Tensor<T> & in )
                : module_(m) , input_(in) {}
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    throw std::runtime_error("ModuleFunctionWrapper forward() should not be called.");
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_output) override{
                    return module_->_internal_backward(grad_output);
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {input_};
                }
        };

        
        template <typename T>
        class NegFunc : public Function<T>{
            private:
                Tensor<T> input;
            public:
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if (inputs.size() != 1){
                        throw std::runtime_error("NegFunc error!");
                    }
                    if(inputs[0].requires_grad()){
                        input = inputs[0];
                    }
                    Tensor<T> result(inputs[0].shape() , inputs[0].device());
                    if (result.device() == Cuda){
                        _neg_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.get() , inputs[0].get() , result.size() );
                    }
                    else{
                        for(int i = 0;i < result.size();i++){
                            result.get()[i] = - inputs[0].get()[i];
                        }
                    }
                    result.set_requires_grad(inputs[0].requires_grad());
                    return std::move(result);
                }
                std::vector<Tensor<T>> backward(const Tensor<T>& grad_output){
                    Tensor<T> gradin(grad_output.shape() , grad_output.device());
                    if(gradin.device() == Cuda){
                        _neg_forward_kernel<<<CudaGetBlocks(gradin.size()) , kCudaThreadsNum>>>(gradin.get() , grad_output.get() , gradin.size());
                    }
                    else{
                        for(int i = 0; i < gradin.size();i++){
                            gradin.get()[i] = -grad_output.get()[i];
                        }
                    }
                    return {gradin};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {input};
                }
        };
        template <typename T>
        __global__ void _add_forward_kernel(T * output, const T* input1, const T* input2 ,  int size){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size){
                output[index] = input1[index] + input2[index];
            }
        }


        template <typename T>
        class AddFunc : public Function<T>{
            private:
                Tensor<T> a , b;
            public:
                AddFunc() = default;
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if (inputs.size() != 2 || inputs[0].shape() != inputs[1].shape()){
                        throw std::runtime_error("AddFunc error!");
                    }
                    if (inputs[0].requires_grad() || inputs[1].requires_grad()){
                        a = inputs[0];
                        b = inputs[1];
                    }
                    if (inputs[0].device() != inputs[1].device()){
                        throw std::runtime_error("AddFunc error!");
                    }
                    Tensor<T> result(inputs[0].shape() , inputs[0].device());
                    if (result.device() == Cuda){
                        _add_forward_kernel<<<CudaGetBlocks(result.size()), kCudaThreadsNum>>>(result.get(), inputs[0].get(), inputs[1].get() , result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.get()[i] = inputs[0].get()[i] + inputs[1].get()[i];
                        }
                    }
                     result.set_requires_grad(inputs[0].requires_grad() || inputs[1].requires_grad());
                    return std::move(result);
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    return {grad_out.deepcopy() , grad_out.deepcopy()};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a , b};
                }
        };
        template <typename T>
        __global__ void _sub_forward_kernel(T * output, const T* input1, const T* input2 ,  int size){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size){
                output[index] = input1[index] - input2[index];
            }
        }
        template <typename T>
        class SubFunc : public Function<T>{
            private:
                Tensor<T> a , b;
            public:
                SubFunc() = default;
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if (inputs.size() != 2 || inputs[0].shape() != inputs[1].shape()){
                        throw std::runtime_error("SubFunc error!");
                    }
                    if (inputs[0].requires_grad() || inputs[1].requires_grad()){
                        a = inputs[0];
                        b = inputs[1];
                    }
                    if (inputs[0].device() != inputs[1].device()){
                        throw std::runtime_error("SubFunc error!");
                    }
                    Tensor<T> result(inputs[0].shape() , inputs[0].device());
                    if (result.device() == Cuda){
                        _sub_forward_kernel<<<CudaGetBlocks(result.size()), kCudaThreadsNum>>>(result.get(), inputs[0].get(), inputs[1].get(),result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.get()[i] = inputs[0].get()[i] - inputs[1].get()[i];
                        }
                    }
                    result.set_requires_grad(inputs[0].requires_grad() || inputs[1].requires_grad());
                    return std::move(result);
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_output) override{
                    return {grad_output.deepcopy() , (-grad_output).deepcopy()};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a , b};
                }
        };

        template <typename T>
        __global__ void _mul_forward_kernel(T * output, const T* input1, const T* input2 ,  int size){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size){
                output[index] = input1[index] * input2[index];
            }
        }

        template <typename T>
        class MulFunc : public Function<T>{
            private:
                Tensor<T> a , b;
            public:

                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if(inputs.size() != 2 || inputs[0].shape() != inputs[1].shape() || inputs[0].device() != inputs[1].device()){
                        throw std::runtime_error("MulFunc error!");
                    }
                    if(inputs[0].requires_grad() || inputs[1].requires_grad()){
                        a = inputs[0] , b = inputs[1];
                    }
                    Tensor<T> result(inputs[0].shape() , inputs[0].device());
                    if(result.device() == Cuda){
                        _mul_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.get() , inputs[0].get() , inputs[1].get(), result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.get()[i] = inputs[0].get()[i] * inputs[1].get()[i];
                        }
                    }
                    result.set_requires_grad(inputs[0].requires_grad() || inputs[1].requires_grad());
                    return std::move(result);
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    Tensor<T> grad_a(grad_out.shape() , grad_out.device());
                    Tensor<T> grad_b(grad_out.shape() , grad_out.device());
                    if (grad_out.device() == Cuda){
                        _mul_forward_kernel<<<CudaGetBlocks(grad_a.size()) , kCudaThreadsNum>>>(grad_a.get() , grad_out.get() , b.get() , grad_a.size());
                        _mul_forward_kernel<<<CudaGetBlocks(grad_b.size()) , kCudaThreadsNum>>>(grad_b.get() , grad_out.get() , a.get() , grad_b.size());
                    }
                    else{
                        for (int i = 0;i<grad_a.size();i++){
                            grad_a.get()[i] = grad_out.get()[i] * b.get()[i];
                            grad_b.get()[i] = grad_out.get()[i] * a.get()[i];
                        }
                    }
                    return {grad_a , grad_b};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a , b};
                }
        };
        template <typename T>
        __global__ void _div_forward_kernel(T * output , const T * input1 , const T * input2 , int size){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size){
                output[index] = input1[index] / input2[index];
            }
        }
        template<typename T>
        __global__ void _div_backward_kernel_1(T * output , const T * grad_out , const T * input , int size){
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index < size){
                output[index] = grad_out[index] / input[index];
            }
        }
        template <typename T>
        __global__ void _div_backward_kernel_2(T * output , const T * grad_out , const T * a , const T * b , int size){
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index < size){
                output[index] = - grad_out[index] * a[index] / (b[index] * b[index]);
            }
        }

        template <typename T>
        class DivFunc : public Function<T>{
            private:
                Tensor<T> a , b;
            public:
                DivFunc() = default;
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if(inputs.size() != 2 || inputs[0].shape() != inputs[1].shape() || inputs[0].device() != inputs[1].device()){
                        throw std::runtime_error("DivFunc error!");
                    }
                    if(inputs[0].requires_grad() || inputs[1].requires_grad()){
                        a = inputs[0] , b = inputs[1];
                    }
                    Tensor<T> result(inputs[0].shape() , inputs[0].device());
                    if(result.device() == Cuda){
                        _div_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.get() , inputs[0].get() , inputs[1].get(), result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.get()[i] = inputs[0].get()[i] / inputs[1].get()[i];
                        }
                    }

                    result.set_requires_grad(inputs[0].requires_grad() || inputs[1].requires_grad());
                    return std::move(result);
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    Tensor<T> grad_a(grad_out.shape() , grad_out.device());
                    Tensor<T> grad_b(grad_out.shape() , grad_out.device());
                    if (grad_out.device() == Cuda){
                        _div_backward_kernel_1<<<CudaGetBlocks(grad_a.size()) , kCudaThreadsNum>>>(grad_a.get() , grad_out.get() , b.get() , grad_a.size());
                        _div_backward_kernel_2<<<CudaGetBlocks(grad_b.size()) , kCudaThreadsNum>>>(grad_b.get() , grad_out.get() , a.get() , b.get() , grad_b.size());
                    }
                    else{
                        for (int i = 0;i<grad_a.size();i++){
                            grad_a.get()[i] = grad_out.get()[i] / b.get()[i];
                            grad_b.get()[i] = - grad_out.get()[i] * a.get()[i] / (b.get()[i] * b.get()[i]);
                        }
                    }
                    return {grad_a , grad_b};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a , b};
                }

        };

        template<typename T>
        __global__ void _relu_forward_kernel(T * output , bool * mask ,  const T * input , int size){
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < size){
                bool ans = input[index] > 0;
                output[index] = ans ? input[index] : 0;
                mask[index] = ans;
            }
        }


        template<typename T>
        __global__ void _relu_backward_kernel(T * grad_in , const T * grad_out , const bool * mask , int size){
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < size){
                grad_in[index] = mask[index] ? grad_out[index] : 0;
            }
        }

        template <typename T>
        class ReLUFunc : public Function<T>{
            private:
                cuda_shared_pointer<bool> mask;
                Tensor<T> a;
            public:
                ReLUFunc() = default;
                Tensor<T> forward(const std::vector<Tensor<T>> & input) override{
                    if(input.size() != 1){
                        throw std::runtime_error("ReLUFunc error!");
                    }
                    if(input[0].requires_grad()){
                        a = input[0];
                    }
                    Tensor<T> result(input[0].shape() , input[0].device());
                    int size = result.size();
                    mask = cuda_shared_pointer<bool>(size , input[0].device());
                    if( input[0].device() == Cuda){
                        _relu_forward_kernel<<<CudaGetBlocks(size) , kCudaThreadsNum>>>(result.get() , mask.get() , input[0].get() , size);
                    }
                    else{
                        for (int i = 0; i < size; i++){
                            result.get()[i] = input[0].get()[i] > 0 ? input[0].get()[i] : 0;
                            mask.get()[i] = input[0].get()[i] > 0;
                        }
                    }
                    result.set_requires_grad(input[0].requires_grad());
                    return std::move(result);
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    Tensor<T> grad_input(grad_out.shape() , grad_out.device());
                    int size = grad_input.size();
                    if(grad_out.device() == Cuda){
                        _relu_backward_kernel<<<CudaGetBlocks(size) , kCudaThreadsNum>>>(grad_input.get() , grad_out.get() , mask.get() , size);
                    }
                    else{
                        for (int i = 0; i < size; i++){
                             grad_input.get()[i] = mask.get()[i] ? grad_out.get()[i] : 0;
                        }
                    }
                    return {std::move(grad_input)};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a};
                }

        };

        template <typename T>
        __global__ void _sigmoid_forward_kernel(T * output , const T * input , int size){
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < size){
                output[index] = 1.0 / (1.0 + nn_exp_device<T>(-input[index]));
            }
        }
        __global__ void _sum_forward_kernel_f(float * output , const float * input ,  const int reduce , const int inner);
        __global__ void _sum_forward_kernel_d(double * output , const double * input ,  const int reduce , const int inner);
        template <typename T>
        __global__ void _sum_backward_kernel(T * grad_in , const T * grad_out , const int reduce , const int inner){
            int ridx = threadIdx.x;
            int iidx = blockIdx.x % inner;
            int oidx = blockIdx.x / inner;
            int ri = reduce * oidx * inner + iidx;
            for(int i = ridx;i<reduce;i+=blockDim.x){
                grad_in[ri + i * inner] = grad_out[oidx * inner];
            }
        }
        template <typename T>
        class SumFunc : public Function<T>{
            private :
                Tensor<T> a;
                int axis;
            public:
                SumFunc( const int axis) : axis(axis){}
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if(inputs.size() != 1){
                        throw std::runtime_error("SumFunc error!");
                    }
                    if(inputs[0].requires_grad()){
                        a = inputs[0];
                    }
                    auto input = inputs[0];
                    axis = input.ndim() - axis - 1;
                    std::vector<int> resultshape;
                    auto inputshape = input.shape();
                    for (int i = 0; i < inputshape.size(); i++){
                        if(i != axis){
                            resultshape.push_back(inputshape[i]);
                        }
                    }
                    if(resultshape.empty()){
                        resultshape.push_back(1);
                    }
                    Tensor<T> result(resultshape , input.device());
                    int indim = input.ndim();
                    int reduce = inputshape[axis];
                    int inner = 1;
                    for(int i = axis + 1;i < indim;i++){
                        inner *= inputshape[i];
                    }
                    if(input.device() == Cuda){
                        if constexpr(std::is_same_v<T , float>){
                            _sum_forward_kernel_f<<< result.size() , kCudaThreadsNum , (kCudaThreadsNum / 32) * sizeof(float)>>>(result.get() , input.get() , reduce , inner);
                        }
                        else if constexpr(std::is_same_v<T , double>){
                            _sum_forward_kernel_d<<< result.size() , kCudaThreadsNum , (kCudaThreadsNum / 32) * sizeof(double)>>>(result.get() , input.get() , reduce , inner);
                        }
                    }
                    else{
                        int outer = result.size() / inner;
                        for(int o = 0;o < outer;o ++ ){
                            for(int i = 0;i < inner; i++){
                                T sum = 0;
                                for(int r = 0;r < reduce;r++){
                                    sum += input.get()[o * reduce * inner + r * inner + i];
                                }
                                result.get()[i + o * inner] = sum;
                            }
                        }
                    }
                    result.set_requires_grad(input.requires_grad());
                    return std::move(result);
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    auto inputshape = a.shape();
                    int indim = inputshape.size();
                    Tensor<T> grad_in(inputshape , a.device());
                    int reduce = inputshape[axis];
                    int inner = 1;
                    if(a.device() == Cuda){
                        for(int i = axis + 1;i < indim;i++){
                            inner *= inputshape[i];
                        }
                        _sum_backward_kernel<<< grad_out.size() , kCudaThreadsNum >>>(grad_in.get() , grad_out.get() , reduce , inner);
                    }
                    else{
                        int outer = grad_out.size() / inner;
                        for(int o = 0;o < outer;o ++ ){
                            for(int i = 0;i < inner; i++){
                                for(int r = 0;r < reduce;r++){
                                    grad_in.get()[o * reduce * inner + r * inner + i] = grad_out.get()[o * inner + i];
                                }
                            }
                        }
                    }
                    return {grad_in};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a};
                }

        };

        template <typename T>
        class SigmoidFunc : public Function<T>{
            private:
                Tensor<T> output;
                Tensor<T> a;
            public:
                SigmoidFunc() = default;
                Tensor<T> forward(const std::vector<Tensor<T>> & input) override{
                    if(input.size() != 1){
                        throw std::runtime_error("SigmoidFunc error!");
                    }
                    if(input[0].requires_grad()){
                        a = input[0];
                    }
                    Tensor<T> result(input[0].shape() , input[0].device());
                    if( input[0].device() == Cuda){
                        _sigmoid_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.get() , input[0].get() , result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.get()[i] = 1 / (1 + nn_exp<T>(-input[0].get()[i]));
                        }
                    }
                    output = result.deepcopy();
                    result.set_requires_grad(input[0].requires_grad());
                    return std::move(result);
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    return {(grad_out * (output * (mytorch::ones<T>(output.shape() , output.device()) - output))).deepcopy()};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a};
                }

        };

        template <typename T>
        std::vector<T> _get_transpose_vec(const std::vector<T> & input , const std::vector<int> & perm){
            std::vector<T> result(input.size());
            for(int i = 0;i < input.size();i++){
                result[i] = input[perm[i]];
            }
            return result;
        }
        template <typename T>
        std::vector<T> _get_transpose_vec_rev(const std::vector<T> & input , const std::vector<int> & perm){
            std::vector<T> result(input.size());
            auto ndim = input.size();
            for(int i = 0;i < input.size();i++){
                result[i] = input[ndim - perm[ndim - i - 1] - 1];
            }
            return result;
        }

        template <typename T>
        std::vector<T> _get_reverse_perm(const std::vector<T> & perm){
            std::vector<T> revperm(perm.size());
            for(int i = 0;i<perm.size();i++){
                revperm[perm[i]] = i;
            }
            return revperm;
        }

        #define divroundup(a , b) ((a + b - 1) / b)
        template <typename T>
        __global__ void _transpose_forward_kernel(T * result ,const T *  input ,const int size ,const int ndim ,const int * inshape 
            ,const int * instrides
            ,const  int * outstrides,
            const int * perm,const int *  revperm){
                extern __shared__ char smem[];
                T * tilem = reinterpret_cast<T *>(smem);
                int threadidx = threadIdx.x , blockidx = blockIdx.x;
                int idx[ kCudaMultiDimMax] , tileidx[ kCudaMultiDimMax];
                bool isvalid = true;
                for(int i = 0;i < ndim;i++){
                    tileidx[i] = blockidx % (divroundup(inshape[ndim - i  - 1] , kCudaTransposeTileSize));
                    idx[i] = threadidx % kCudaTransposeTileSize;
                    if(tileidx[i] * kCudaTransposeTileSize + idx[i] >= inshape[ndim -i - 1]){
                        isvalid = false;
                    }
                    blockidx /= divroundup(inshape[ndim - i - 1] , kCudaTransposeTileSize);
                    threadidx /= kCudaTransposeTileSize;
                }
                int index = 0;
                for(int i = 0;i< ndim;i++){
                    index += (idx[i] + kCudaTransposeTileSize * tileidx[i]) * instrides[i];
                }
                if (isvalid){
                    tilem[threadIdx.x] = input[index];
                }
                __syncthreads();
                int outputindex = 0;
                int outtileindex = 0;
                isvalid = true;
                for(int i = 0;i<ndim;i++){
                    outputindex += (idx[i] + kCudaTransposeTileSize * tileidx[ndim - perm[ndim - i - 1] - 1]) * outstrides[i];
                    if(idx[i] + kCudaTransposeTileSize * tileidx[ndim - perm[ndim - i - 1] - 1] >= inshape[perm[ndim -i - 1]]){
                        isvalid = false;
                        break;
                    }
                    outtileindex += idx[ndim - revperm[i] - 1];
                    outtileindex *= kCudaTransposeTileSize;
                }
                outtileindex /= kCudaTransposeTileSize;
                if(isvalid){
                    result[outputindex] = tilem[outtileindex];
                }
        }


        template <typename T>
        __global__ void _transpose_forward_kernel_2dim(T * result ,const T *  input  , const int m, const int n){
            extern __shared__ char smem[];
            T * tilem = reinterpret_cast<T *>(smem);
            int idx[2];
            idx[0] = threadIdx.x + kCudaTransposeTileSize * blockIdx.x; // n
            idx[1] = threadIdx.y + kCudaTransposeTileSize * blockIdx.y; // m
            int index = idx[0] + idx[1] * n;
            if(idx[0] < n && idx[1] < m){
                tilem[threadIdx.y + threadIdx.x * kCudaTransposeTileSize] = input[index];
            }
            __syncthreads();
            idx[0] = threadIdx.x + kCudaTransposeTileSize * blockIdx.y; // m
            idx[1] = threadIdx.y + kCudaTransposeTileSize * blockIdx.x; // n
            index = idx[0] + idx[1] * m;
            if(idx[0] < m && idx[1] < n){
                result[index] = tilem[threadIdx.y * kCudaTransposeTileSize + threadIdx.x];
            }
        }

        template <typename T>
        __global__ void _transpose_forward_kernel_HWC2CHW(T * result, const T * input , const int h , const int w , const int c){
            extern __shared__ char smem[];
            T * tilem = reinterpret_cast<T *>(smem);
            int idx[3];
            idx[0] = threadIdx.x + kCudaTransposeTileSize * blockIdx.x; // c
            idx[1] = threadIdx.y + kCudaTransposeTileSize * blockIdx.y; // w
            idx[2] = threadIdx.z + kCudaTransposeTileSize * blockIdx.z; // h
            int index = idx[0] + idx[1] * c + idx[2] * w * c;
            if(idx[0] < c && idx[1] < w && idx[2] < h){
                tilem[threadIdx.y + threadIdx.z * kCudaTransposeTileSize + threadIdx.x * kCudaTransposeTileSize * kCudaTransposeTileSize] = input[index];
            }
            __syncthreads();
            idx[0] = threadIdx.x + blockIdx.y * kCudaTransposeTileSize; // w
            idx[1] = threadIdx.y + blockIdx.z * kCudaTransposeTileSize; // h
            idx[2] = threadIdx.z + blockIdx.x * kCudaTransposeTileSize; // c
            index = idx[0] + w * idx[1] + w * h * idx[2];
            if(idx[0] < w && idx[1] < h && idx[2] < c){
                result[index] = tilem[threadIdx.x + threadIdx.y * kCudaTransposeTileSize + threadIdx.z * kCudaTransposeTileSize * kCudaTransposeTileSize];
            }

        }
        template <typename T>
        __global__ void _transpose_forward_kernel_CHW2HWC(T * result, const T * input , const int c , const int h , const int w){
            extern __shared__ char smem[];
            T * tilem = reinterpret_cast<T *>(smem);
            int idx[3];
            idx[0] = threadIdx.x + kCudaTransposeTileSize * blockIdx.x; // w
            idx[1] = threadIdx.y + kCudaTransposeTileSize * blockIdx.y; // h
            idx[2] = threadIdx.z + kCudaTransposeTileSize * blockIdx.z; // c
            int index = idx[0] + idx[1] * w + idx[2] * w * h;
            if(idx[0] < w && idx[1] < h && idx[2] < c){
                tilem[threadIdx.z + threadIdx.x * kCudaTransposeTileSize + threadIdx.y * kCudaTransposeTileSize * kCudaTransposeTileSize] = input[index];
            }
            __syncthreads();
            idx[0] = threadIdx.x + blockIdx.z * kCudaTransposeTileSize; // c
            idx[1] = threadIdx.y + blockIdx.x * kCudaTransposeTileSize; // w
            idx[2] = threadIdx.z + blockIdx.y * kCudaTransposeTileSize; // h
            index = idx[0] + idx[1] * c + idx[2] * c * w;
            if(idx[0] < c && idx[1] < w && idx[2] < h){
                result[index] = tilem[threadIdx.x + threadIdx.y * kCudaTransposeTileSize + threadIdx.z * kCudaTransposeTileSize * kCudaTransposeTileSize];
            }

        }




        template <typename T>
        class TransposeFunc : public Function<T>{
            private:
                std::vector<int> perm;
                Tensor<T> a;
                enum TransposeType{
                    TLast2Dim,
                    THWC2CHW,
                    TCHW2HWC,
                    TGeneral
                };
                TransposeType ttype;
                cudaStream_t streams[8];

            public:
                const int streamCount = 4;
                ~TransposeFunc(){
                    for(int i = 0;i<streamCount;i++){
                        CHECK(cudaStreamDestroy(streams[i]));
                    }
                }
                
                TransposeFunc(const std::vector<int> & perm) : perm(perm){
                    for(int i = 0;i<streamCount;i++){
                        CHECK(cudaStreamCreate(&streams[i]));
                    }
                    int eq_count = 0;
                    for(eq_count = 0;eq_count<perm.size() ;eq_count++){
                        if(perm[eq_count] != eq_count){
                            break;
                        }
                    }
                    if(eq_count == perm.size() - 2 && perm.back() == perm.size() - 2){
                        ttype = TLast2Dim;
                    }
                    else if(eq_count == perm.size() - 3 && perm.back() == perm.size() - 2 && perm[eq_count] == perm.size() - 1){
                        ttype = THWC2CHW;
                    }
                    else if(eq_count == perm.size() - 3 && perm.back() == eq_count && perm[eq_count] == perm.size() - 2){
                        ttype = TCHW2HWC;
                    }
                    else{
                        ttype = TGeneral;
                    }
                }
                Tensor<T> forward(const std::vector<Tensor<T>> & input ) override{
                    if(input.size() != 1)
                        throw std::runtime_error("TransposeFunc error!");
                    if(input[0].requires_grad()){
                        a = input[0];
                    }
                    std::vector<int> newshape = _get_transpose_vec(input[0].shape() , perm);
                    Tensor<T> result(newshape , input[0].device());
                    result.set_requires_grad(input[0].requires_grad());
                    if(result.device() == Cuda){
                        if(ttype == TLast2Dim){


                            int ndim = input[0].shape().size();
                            int m = input[0].shape()[ndim - 2];
                            int n = input[0].shape()[ndim - 1];
                            int batchstep = m * n;
                            int batch_size = input[0].size()/ batchstep;

                            dim3 grid_size(divroundup(n , kCudaTransposeTileSize) , divroundup(m , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize);


                            for(int b = 0;b < batch_size;b ++){
                                int s = b % streamCount;
                                _transpose_forward_kernel_2dim
                                <<<grid_size , block_size , sizeof(T) * kCudaTransposeTileSize * kCudaTransposeTileSize , streams[s]>>>
                                (
                                    result.get() + b * batchstep,
                                    input[0].get() + b * batchstep,
                                    m , n
                                );

                            }
                        }
                        else if(ttype == THWC2CHW){


                            int ndim = input[0].shape().size();
                            int h = input[0].shape()[ndim - 3];
                            int w = input[0].shape()[ndim - 2];
                            int c = input[0].shape()[ndim - 1];
                            int batchstep = h * w * c;
                            int batch_size = input[0].size()/ batchstep;

                            dim3 grid_size(divroundup(c , kCudaTransposeTileSize) , divroundup(w , kCudaTransposeTileSize) , divroundup(h , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize , kCudaTransposeTileSize );


                            for(int b = 0;b < batch_size;b ++){
                                int s = b % streamCount;
                                _transpose_forward_kernel_HWC2CHW
                                <<<grid_size , block_size , sizeof(T) * kCudaTransposeTileSize * kCudaTransposeTileSize * kCudaTransposeTileSize , streams[s]>>>
                                (
                                    result.get() + b * batchstep,
                                    input[0].get() + b * batchstep,
                                    h , w , c
                                );

                            }
                        }
                        else if(ttype == TCHW2HWC){


                            int ndim = input[0].shape().size();
                            int c = input[0].shape()[ndim - 3];
                            int h = input[0].shape()[ndim - 2];
                            int w = input[0].shape()[ndim - 1];
                            int batchstep = h * w * c;
                            int batch_size = input[0].size()/ batchstep;

                            dim3 grid_size(divroundup(w , kCudaTransposeTileSize) , divroundup(h , kCudaTransposeTileSize) , divroundup(c , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize , kCudaTransposeTileSize );


                            for(int b = 0;b < batch_size;b ++){
                                int s = b % streamCount;
                                _transpose_forward_kernel_CHW2HWC
                                <<<grid_size , block_size , sizeof(T) * kCudaTransposeTileSize * kCudaTransposeTileSize * kCudaTransposeTileSize , streams[s]>>>
                                (
                                    result.get() + b * batchstep,
                                    input[0].get() + b * batchstep,
                                    c , h , w
                                );

                            }
                        }
                        else{
                            std::vector<int> revperm = _get_reverse_perm(perm);
                            int totalthreads = 1;
                            for(int i = 0;i < input[0].shape().size();i++){
                                totalthreads *= divroundup(input[0].shape()[i] , kCudaTransposeTileSize);
                            }
                            int tilesize = (1 << (2 * input[0].shape().size()));
                            cuda_shared_pointer<int> shape(input[0].shape() , Cuda);
                            cuda_shared_pointer<int> outstrides(result.get_strides() , Cuda);
                            cuda_shared_pointer<int> instrides(input[0].get_strides() , Cuda);
                            cuda_shared_pointer<int> cuperm(perm , Cuda);
                            cuda_shared_pointer<int> curevperm(revperm , Cuda);

                            _transpose_forward_kernel<<<totalthreads , tilesize , sizeof(T) * tilesize>>>(result.get() , input[0].get() , 
                                result.size() , shape.size() , shape.get() , instrides.get() , outstrides.get() , cuperm.get() , curevperm.get());


                            }
                        return std::move(result);
                    }
                    else{
                        auto instrides = input[0].get_strides();
                        instrides.push_back(input[0].size());
                        auto strides = result.get_strides();
                        int ndim = input[0].shape().size();
                        for(int index = 0;index<input[0].size();index+= instrides[0]){
                            std::vector<int> idx(ndim);
                            for(int i = 0;i<ndim;i++){
                                idx[i] = index % instrides[i+1] / instrides[i];
                            }
                            std::vector<int> outidx = _get_transpose_vec_rev(idx , perm);
                            int outindex = 0;
                            for(int i = 0;i<ndim;i++){
                                outindex += outidx[i] * strides[i];
                            }
                            result.get()[outindex] = input[0].get()[index];
                        }
                    }

                    return std::move(result);
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out){
                    //return {grad_out.transpose(_get_reverse_perm(perm))};
                    std::vector<int> revperm = _get_reverse_perm(perm);
                    std::vector<int> newshape = a.shape();
                    Tensor<T> result(newshape , a.device());
                    if(result.device() == Cuda){
                        TransposeType ttype;
                        int eq_count = 0;
                        for(eq_count = 0;eq_count<revperm.size() ;eq_count++){
                            if(revperm[eq_count] != eq_count){
                                break;
                            }
                        }
                        if(eq_count == revperm.size() - 2 && revperm.back() == revperm.size() - 2){
                            ttype = TLast2Dim;
                        }
                        else if(eq_count == revperm.size() - 3 && revperm.back() == revperm.size() - 2 && revperm[eq_count] == revperm.size() - 1){
                            ttype = THWC2CHW;
                        }
                        else if(eq_count == revperm.size() - 3 && revperm.back() == eq_count && revperm[eq_count] == revperm.size() - 2){
                            ttype = TCHW2HWC;
                        }
                        else{
                            ttype = TGeneral;
                        }

                        if(ttype == TLast2Dim){


                            int ndim = grad_out.shape().size();
                            int m = grad_out.shape()[ndim - 2];
                            int n = grad_out.shape()[ndim - 1];
                            int batchstep = m * n;
                            int batch_size = grad_out.size()/ batchstep;

                            dim3 grid_size(divroundup(n , kCudaTransposeTileSize) , divroundup(m , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize);


                            for(int b = 0;b < batch_size;b ++){
                                int s = b % streamCount;
                                _transpose_forward_kernel_2dim
                                <<<grid_size , block_size , sizeof(T) * kCudaTransposeTileSize * kCudaTransposeTileSize , streams[s]>>>
                                (
                                    result.get() + b * batchstep,
                                    grad_out.get() + b * batchstep,
                                    m , n
                                );

                            }
                        }
                        else if(ttype == THWC2CHW){

                            int ndim = grad_out.shape().size();
                            int h = grad_out.shape()[ndim - 3];
                            int w = grad_out.shape()[ndim - 2];
                            int c = grad_out.shape()[ndim - 1];
                            int batchstep = h * w * c;
                            int batch_size = grad_out.size()/ batchstep;

                            dim3 grid_size(divroundup(c , kCudaTransposeTileSize) , divroundup(w , kCudaTransposeTileSize) , divroundup(h , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize , kCudaTransposeTileSize );


                            for(int b = 0;b < batch_size;b ++){
                                int s = b % streamCount;
                                _transpose_forward_kernel_HWC2CHW
                                <<<grid_size , block_size , sizeof(T) * kCudaTransposeTileSize * kCudaTransposeTileSize * kCudaTransposeTileSize , streams[s]>>>
                                (
                                    result.get() + b * batchstep,
                                    grad_out.get() + b * batchstep,
                                    h , w , c
                                );

                            }
                        }
                        else if(ttype == TCHW2HWC){

                            int ndim = grad_out.shape().size();
                            int c = grad_out.shape()[ndim - 3];
                            int h = grad_out.shape()[ndim - 2];
                            int w = grad_out.shape()[ndim - 1];
                            int batchstep = h * w * c;
                            int batch_size = grad_out.size()/ batchstep;

                            dim3 grid_size(divroundup(w , kCudaTransposeTileSize) , divroundup(h , kCudaTransposeTileSize) , divroundup(c , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize , kCudaTransposeTileSize );


                            for(int b = 0;b < batch_size;b ++){
                                int s = b % streamCount;
                                _transpose_forward_kernel_CHW2HWC
                                <<<grid_size , block_size , sizeof(T) * kCudaTransposeTileSize * kCudaTransposeTileSize * kCudaTransposeTileSize , streams[s]>>>
                                (
                                    result.get() + b * batchstep,
                                    grad_out.get() + b * batchstep,
                                    c , h , w
                                );

                            }
                        }
                        else{
                             int totalthreads = 1;
                            for(int i = 0;i < grad_out.shape().size();i++){
                                totalthreads *= divroundup(grad_out.shape()[i] , kCudaTransposeTileSize);
                            }
                            int tilesize = (1 << (2 * grad_out.shape().size()));
                            cuda_shared_pointer<int> shape(grad_out.shape() , Cuda);
                            cuda_shared_pointer<int> outstrides(result.get_strides() , Cuda);
                            cuda_shared_pointer<int> instrides(grad_out.get_strides() , Cuda);
                            cuda_shared_pointer<int> cuperm(revperm , Cuda);
                            cuda_shared_pointer<int> curevperm(perm , Cuda);

                            _transpose_forward_kernel<<<totalthreads , tilesize , sizeof(T) * tilesize>>>(result.get() , grad_out.get() , 
                                result.size() , shape.size() , shape.get() , instrides.get() , outstrides.get() , cuperm.get() , curevperm.get());


                            }
                        return {std::move(result)};
                    }
                    else{
                        auto instrides = grad_out.get_strides();
                        instrides.push_back(grad_out.size());
                        auto strides = result.get_strides();
                        int ndim = grad_out.shape().size();
                        for(int index = 0;index<grad_out.size();index+= instrides[0]){
                            std::vector<int> idx(ndim);
                            for(int i = 0;i<ndim;i++){
                                idx[i] = index % instrides[i+1] / instrides[i];
                            }
                            std::vector<int> outidx = _get_transpose_vec_rev(idx , revperm);
                            int outindex = 0;
                            for(int i = 0;i<ndim;i++){
                                outindex += outidx[i] * strides[i];
                            }
                            result.get()[outindex] = grad_out.get()[index];
                        }
                    }

                    return {std::move(result)};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a};
                }



        };


        template <typename T>
        __global__ void _pool_forward_kernel(T * result , const T * input , int * mask ,int ndim ,  const int * kernel_shape , 
            const int result_size ,  const int *  input_shape , const int * result_shape
             , const int * input_strides){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if(index >= result_size)
                return;
            int outidx[kCudaMultiDimMax];
            int index_ = index;
            for(int i = 0;i<ndim;i++){
                outidx[i] = index_ % result_shape[ndim - i - 1];
                index_ /= result_shape[ndim - i - 1];
            }
            CudaMultiDimIndex kernel_idx(kernel_shape , ndim);
            do{
                int inputindex = 0;
                for(int i = 0;i<ndim;i++){
                    inputindex += ((outidx[i] * kernel_shape[ndim - i - 1]) + kernel_idx[ndim - i - 1]) * input_strides[i];
                }
                if(result[index] < input[inputindex]){
                    result[index] = input[inputindex];
                    mask[index] = inputindex;
                }
                kernel_idx.next();
            }while(!kernel_idx.is_zero());
        }

        template <typename T>
        __global__ void _maxpool2d_forward_kernel(T * result , const T * input , int * mask  ,  const int kh , const int kw , 
            const int result_size ,  const int h , const int w , const int rh , const int rw){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if(index >= result_size)
                return;
            int rhw = rh * rw;
            int bid = index / rhw;
            int r0 = (index / rw) % rh;
            int r1 = index % rw;
            int m0 = r0 * kh;
            int m1 = r1 * kw;
            int offset = m0 * w + m1 + h * w * bid;
            T maxval = -FLT_MAX;
            int maxindex;
            for(int i = 0;i < kh ; i++){
                for(int j = 0;j < kw;j++){
                    int curindex = offset + i * w + j;
                    if(maxval < input[curindex]){
                        maxval = input[curindex];
                        maxindex = curindex;
                    }
                }
            }
            result[index] = maxval;
            mask[index] = maxindex;
        }
        template <typename T>
        __global__ void _pool_backward_kernel(T * grad_in , const T * grad_out ,const int * mask , 
            const int result_size ){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if(index >= result_size)
                return;
            int inputindex = mask[index];
            grad_in[inputindex] = grad_out[index];
        }

        template <typename T>
        class MaxPool2dFunc : public Function<T>{
            private:
                int kh , kw;
                int * mask;
                int mask_size;
                Tensor<T> a;
                Device device;
            public:
                MaxPool2dFunc(const std::vector<int> & kernel_shape , Device device) : kh(kernel_shape[0]) , kw(kernel_shape[1]) , device(device){ 
                    mask = 0;
                    mask_size = 0;
                }
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs){
                    auto input = inputs[0];
                    if(inputs.size() != 1){
                        std::cerr << "MaxPool2dFunc error! input size must be 1, but got " << inputs.size() << std::endl;
                        throw std::runtime_error("PoolFunc error!");
                    }
                    if(input.device() != device){
                        std::cerr << "MaxPool2dFunc error! input device must be the same." << std::endl;
                        throw std::runtime_error("PoolFunc error!");
                    }
                    auto inputshape = input.shape();
                    int ndim = input.ndim();
                    int h = inputshape[ndim - 2];
                    int w = inputshape[ndim - 1];
                    int rh = h / kh;
                    int rw = w / kw;
                    std::vector<int> resultshape = inputshape;
                    resultshape[ndim - 2] = rh;
                    resultshape[ndim - 1] = rw;
                    Tensor<T> result(resultshape , false , device);
                    if(device == Cuda){
                        if(mask_size < result.size() * sizeof(int)){
                            if(mask){
                                CHECK(cudaFree(mask));
                            }
                            mask_size = result.size() * sizeof(int);
                            CHECK(cudaMalloc(&mask , mask_size));
                        }
                        _maxpool2d_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(
                            result.get(),
                            input.get(),
                            mask,
                            kh,
                            kw,
                            result.size(),
                            h,
                            w,
                            rh,rw
                        );
                    }
                    else{
                        if(mask_size < result.size() * sizeof(int)){
                            if(mask){
                                delete [] mask;
                            }
                            mask = new int[result.size()];
                            mask_size = result.size() * sizeof(int);
                        }
                        int resultbatchstep = rh * rw , inputbatchstep = h * w;
                        for(int resultbatchoffset = 0 , inputbatchoffset = 0;
                            resultbatchoffset < result.size();
                            resultbatchoffset += resultbatchstep , inputbatchoffset += inputbatchstep){
                            for(int r0 = 0;r0 < rh;r0++){
                                for(int r1 = 0;r1 < rw;r1++){
                                    T maxval = -FLT_MAX;
                                    int maxindex;
                                    int rindex = resultbatchoffset + r0 * rw + r1;
                                    for(int k0 = 0;k0 < kh;k0++){
                                        for(int k1 = 0;k1 < kw;k1++){
                                            int iindex = inputbatchoffset + ((r0 * kh) + k0) * w + (r1 * kw) + k1;
                                            T val = input[iindex];
                                            if(maxval < val){
                                                maxval = val;
                                                maxindex = iindex;
                                            }
                                        }
                                    }
                                    mask[rindex] = maxindex;
                                    result[rindex] = maxval;
                                }
                            }
                        }

                    }
                    if(input.requires_grad()){
                        a = input;
                        result.set_requires_grad(true);
                        }
                    return std::move(result);
                }

                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) {
                    auto input = a;
                    Tensor<T> grad_in(input.shape() , device);
                    if(device == Cuda){
                        _pool_backward_kernel<<<CudaGetBlocks(grad_out.size()) , kCudaThreadsNum>>>(
                            grad_in.get() , grad_out.get() , mask , grad_out.size()
                        );
                    }
                    else{
                        for(int i = 0;i < grad_out.size();i++){
                            int index = mask[i];
                            grad_in.get()[index] += grad_out.get()[i];
                        }
                    }
                    return {std::move(grad_in)};
                }
                std::vector<Tensor<T>> get_inputs() const{
                    return {a};
                }

        };
        template <typename T>
        class ReshapeFunc : public Function<T>{
            private:
                std::vector<int> newshape;
                std::vector<int> oldshape;
                Tensor<T> a;
            public:
                ReshapeFunc(const std::vector<int> & newshape) : newshape(newshape){}
                Tensor<T> forward(const std::vector<Tensor<T>> & input ) override{
                    oldshape = input[0].shape();
                    if(input.size() != 1 )
                        throw std::runtime_error("ReshapeFunc error!");
                    if(input[0].requires_grad()){
                        a = input[0];
                    }
                    Tensor<T> result = make_view<T>(input[0].get_shared_ptr() , newshape);
                    result.set_requires_grad(input[0].requires_grad());
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out){
                    return {grad_out.reshape(oldshape)};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a};
                }

        };

        template <>
        class MatmulFunc<float> : public Function<float>{
            private:
                Tensor<float> a; 
                Tensor<float> b;
            public:
                MatmulFunc() = default;
                Tensor<float> forward(const std::vector<Tensor<float>> & input) override{
                    if(input.size() != 2 || input[0].shape().size() != input[1].shape().size() 
                        || input[0].shape().size() < 2 
                        || input[0].shape()[input[0].shape().size() - 1] != input[1].shape()[input[1].shape().size() - 2] || 
                        input[0].get_strides()[0] != 1
                        || input[1].get_strides()[0] != 1)
                        throw std::runtime_error("MatmulFunc error!");
                    if(input[0].requires_grad() || input[1].requires_grad()){
                        a = input[0];
                        b = input[1];
                    }
                    std::vector<int> newshape = input[0].shape();
                    newshape[newshape.size() - 1] = input[1].shape()[input[1].shape().size() - 1];
                    Tensor<float> result(newshape , input[0].device());
                    result.set_requires_grad(input[0].requires_grad() || input[1].requires_grad());
                    auto resultshape = result.shape();
                    auto input0shape = input[0].shape();
                    auto input1shape = input[1].shape();
                    auto input0stride = input[0].get_strides();
                    input0stride.push_back(input[0].size());
                    auto input1stride = input[1].get_strides();
                    input1stride.push_back(input[1].size());
                    auto resultstride = result.get_strides();
                    resultstride.push_back(result.size());
                    int step0 = input0stride[2];
                    int step1 = input1stride[2];
                    int stepresult = resultstride[2];
                    if(result.device() == Cuda){
                        cublasHandle_t handle;
                        cublasCreate(&handle);
                        float alpha = 1.0f;
                        float beta = 0.0f;
                        CHECK_CUBLAS(cublasSgemmStridedBatched(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                            resultshape[resultshape.size() - 1] , 
                            resultshape[resultshape.size() - 2],
                            input0shape[input0shape.size() - 1] , 
                            &alpha , 
                            input[1].get()  , 
                            input1stride[1] , 
                            step1,
                            input[0].get()  , 
                            input0stride[1] , 
                            step0,
                            &beta , 
                            result.get()   , 
                            resultstride[1],
                            stepresult,
                            result.size() / stepresult
                        ));
                        CHECK_CUBLAS(cublasDestroy(handle));
                    }
                    else{
                        for(int offset0 = 0 , offset1 = 0 , offsetresult = 0;offsetresult < result.size();offset0 += step0 , offset1 += step1 , offsetresult += stepresult){
                            for(int i = 0;i<result.shape()[result.shape().size() - 2];i++){
                                for(int j = 0;j<result.shape()[result.shape().size() - 1];j++){
                                    float sum = 0.0;
                                    for(int k = 0;k<input0shape[input0shape.size() - 1];k++){
                                        sum += input[0].get()[offset0 + i * input0stride[1] + k * input0stride[0]] 
                                            * input[1].get()[offset1 + k * input1stride[1] + j * input1stride[0]];
                                    }
                                    result.get()[offsetresult + i * resultstride[1] + j * resultstride[0]] = sum;
                                }
                            }
                        }
                    }
                    return result;
                }
                std::vector<Tensor<float>> backward(const Tensor<float> & grad_out){
                    std::vector<Tensor<float>> result;
                    std::vector<int> gradperm;
                    int ndim =  a.shape().size();
                    for(int i = 0;i<ndim ;i++){
                        gradperm.push_back(i);
                    }
                    std::swap(gradperm[ndim - 2] , gradperm[ndim - 1]);
                    result.push_back(grad_out.matmul(b.transpose(gradperm)));
                    result.push_back(a.transpose(gradperm).matmul(grad_out));
                    return result;
                }
                std::vector<Tensor<float>> get_inputs() const override{
                    return {std::move(a) , std::move(b)};
                }
                
        };
        template <>
        class MatmulFunc<double> : public Function<double>{
            private:
                Tensor<double> a; 
                Tensor<double> b;
            public:
                MatmulFunc() = default;
                Tensor<double> forward(const std::vector<Tensor<double>> & input) override{
                    if(input.size() != 2 || input[0].shape().size() != input[1].shape().size() 
                        || input[0].shape().size() < 2 
                        || input[0].shape()[input[0].shape().size() - 1] != input[1].shape()[input[1].shape().size() - 2] || 
                        input[0].get_strides()[0] != 1
                        || input[1].get_strides()[0] != 1)
                        throw std::runtime_error("MatmulFunc error!");
                    if(input[0].requires_grad() || input[1].requires_grad()){
                        a = input[0];
                        b = input[1];
                    }
                    std::vector<int> newshape = input[0].shape();
                    newshape[newshape.size() - 1] = input[1].shape()[input[1].shape().size() - 1];
                    Tensor<double> result(newshape , input[0].device());
                    result.set_requires_grad(input[0].requires_grad() || input[1].requires_grad());
                    auto resultshape = result.shape();
                    auto input0shape = input[0].shape();
                    auto input1shape = input[1].shape();
                    auto input0stride = input[0].get_strides();
                    input0stride.push_back(input[0].size());
                    auto input1stride = input[1].get_strides();
                    input1stride.push_back(input[1].size());
                    auto resultstride = result.get_strides();
                    resultstride.push_back(result.size());
                    int step0 = input0stride[2];
                    int step1 = input1stride[2];
                    int stepresult = resultstride[2];
                    if(result.device() == Cuda){
                        cublasHandle_t handle;
                        cublasCreate(&handle);
                        double alpha = 1.0f;
                        double beta = 0.0f;
                        CHECK_CUBLAS(cublasDgemmStridedBatched(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                            resultshape[resultshape.size() - 1] , 
                            resultshape[resultshape.size() - 2],
                            input0shape[input0shape.size() - 1] , 
                            &alpha , 
                            input[1].get()  , 
                            input1stride[1] , 
                            step1,
                            input[0].get()  , 
                            input0stride[1] , 
                            step0,
                            &beta , 
                            result.get()   , 
                            resultstride[1],
                            stepresult,
                            result.size() / stepresult
                        ));
                        CHECK_CUBLAS(cublasDestroy(handle));
                    }
                    else{
                        for(int offset0 = 0 , offset1 = 0 , offsetresult = 0;offsetresult < result.size();offset0 += step0 , offset1 += step1 , offsetresult += stepresult){
                            for(int i = 0;i<result.shape()[result.shape().size() - 2];i++){
                                for(int j = 0;j<result.shape()[result.shape().size() - 1];j++){
                                    double sum = 0.0;
                                    for(int k = 0;k<input0shape[input0shape.size() - 1];k++){
                                        sum += input[0].get()[offset0 + i * input0stride[1] + k * input0stride[0]] 
                                            * input[1].get()[offset1 + k * input1stride[1] + j * input1stride[0]];
                                    }
                                    result.get()[offsetresult + i * resultstride[1] + j * resultstride[0]] = sum;
                                }
                            }
                        }
                    }
                    return result;
                }
                std::vector<Tensor<double>> backward(const Tensor<double> & grad_out){
                    std::vector<Tensor<double>> result;
                    std::vector<int> gradperm;
                    int ndim =  a.shape().size();
                    for(int i = 0;i<ndim ;i++){
                        gradperm.push_back(i);
                    }
                    std::swap(gradperm[ndim - 2] , gradperm[ndim - 1]);
                    result.push_back(grad_out.matmul(b.transpose(gradperm)));
                    result.push_back(a.transpose(gradperm).matmul(grad_out));
                    return result;
                }
                std::vector<Tensor<double>> get_inputs() const override{
                    return {a , b};
                }
                
        };
        template <typename T>
        T prod_vec(const std::vector<T> & vec){
            T result = 1.0;
            for(int i = 0;i<vec.size();i++){
                result *= vec[i];
            }
            return result;
        }

        template <typename T>
        T dot_vec(const std::vector<T> & a , const std::vector<T> & b){
            T sum = 0;
            for(int i = 0;i<a.size();i++){
                sum += a[i] * b[i];
            }
            return sum;
        }
        template <typename T>
        T rev_dot_vec(const std::vector<T> & a , const std::vector<T> & b){
            T sum = 0;
            for(int i = 0;i<a.size();i++){
                sum += a[i] * b[a.size() - 1 - i];
            }
            return sum;
        }


    }
    template <typename T>
    void __cudaMemcpyBatch(T * dst , const T * src ,const int size , const int batch_size , cudaStream_t * streams ){

        for(int b = 0;b < batch_size;b ++){
            int s = b % kStreamCount;
            cudaMemcpyAsync(dst + b * size, src , size * sizeof(T) , cudaMemcpyDeviceToDevice , streams[s]);

        }
    }
    template <typename T>
    __global__ void _linear_add(T * output , const T * input , int size){
        int index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < size){
            output[index] += input[index];
        }
    }
    template <typename T>
    class Linear : public Module<T>{
        private:
            Tensor<T> input_cache; // internal backward
            cudaStream_t stream_gemm,stream_add;
            cublasHandle_t handle;
            cudaStream_t streams[kStreamCount];
            Tensor<T> weight_t;
            bool optimal =false;
        public:
            Tensor<T> weight;
            Tensor<T> bias;
            Linear(const Tensor<T> & weight, const Tensor<T> & bias)
            : weight(weight), bias(bias) {
                if(weight.ndim() != 2){
                    std::cerr << "Linear: weight must be 2D tensor" << std::endl;
                    throw std::runtime_error("Linear: weight must be 2D tensor");
                }
                if(bias.ndim() != 1){
                    std::cerr << "Linear: bias must be 1D tensor" << std::endl;
                    throw std::runtime_error("Linear: bias must be 1D tensor");
                }
                if(weight.shape()[0] != bias.size()){
                    std::cerr << "Linear: weight shape and bias shape mismatch" << std::endl;
                    throw std::runtime_error("Linear: weight shape and bias shape mismatch");
                }
                if(weight.device() == Cuda){
                    cublasCreate(&handle);
                    CHECK(cudaStreamCreate(&stream_gemm));
                    CHECK(cudaStreamCreate(&stream_add));
                    for(int i = 0;i<kStreamCount;i++){
                        CHECK(cudaStreamCreate(&streams[i]));
                    }

                }
            }
            Linear(const int in_features, const int out_features, Device device = DefaultDevice){
                T inv = nn_rsqrt<T>(in_features);
                weight = rand<T>({out_features , in_features} , device) * 2 * inv - inv;
                bias = rand<T>({out_features} , device) * 2 * inv - inv;
                if(device == Cuda){
                    cublasCreate(&handle);
                    CHECK(cudaStreamCreate(&stream_gemm));
                    CHECK(cudaStreamCreate(&stream_add));
                    for(int i = 0;i<kStreamCount;i++){
                        CHECK(cudaStreamCreate(&streams[i]));
                    }
                }
            }
            ~Linear(){
                if(weight.device() == Cuda){
                    cublasDestroy(handle);
                    CHECK(cudaStreamDestroy(stream_gemm));
                    CHECK(cudaStreamDestroy(stream_add));
                    for(int i = 0;i<kStreamCount;i++){
                        CHECK(cudaStreamDestroy(streams[i]));
                    }
                }
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                weight_t = weight.transpose({1 , 0});
                if(inputs.size() != 1){
                    throw std::runtime_error("Linear: input size must be 1");
                }
                auto input = inputs[0];
                if(input.shape().back() != weight.shape().back()){
                    std::cerr << "Linear: input shape and weight shape mismatch" << std::endl;
                    throw std::runtime_error("Linear: input shape and weight shape mismatch");
                }
                if(input.requires_grad()){
                    input_cache = input;
                }
                std::vector<int> newshape = input.shape();
                newshape.pop_back();
                newshape.push_back(bias.size());
                Tensor<T> result(newshape , false, input.device()  );
                result.set_requires_grad(input.requires_grad());
                auto resultshape = result.shape();
                auto input0shape = weight.shape();
                auto input1shape = input.shape();
                auto input0stride = weight.get_strides();
                input0stride.push_back(weight.size());
                auto input1stride = input.get_strides();
                input1stride.push_back(input.size());
                auto resultstride = result.get_strides();
                resultstride.push_back(result.size());
                int step1 = input1stride[1];
                int stepresult = resultstride[1];


                T * resultget = result.get();
                const T * inputget = input.get();


                if(result.device() == Cuda){
                    T alpha = 1.0f;
                    T beta0 = 1.0f;
                    __cudaMemcpyBatch(resultget , bias.get() , stepresult , result.size() / stepresult , streams);

                    auto batch_size = result.size() / stepresult;

                    if constexpr (std::is_same_v<T , float>){
                        CHECK_CUBLAS(cublasSgemvStridedBatched(
                            handle,
                            CUBLAS_OP_N,
                            resultshape[resultshape.size() - 1],
                            input0shape[input0shape.size() - 1],
                            &alpha,
                            weight_t.get(),
                            weight_t.shape().back(),
                            0,
                            inputget,
                            1,
                            step1,
                            &beta0,
                            resultget,
                            1,
                            stepresult,
                            batch_size
                        ));
                    }
                    else if constexpr (std::is_same_v<T , double>){
                        CHECK_CUBLAS(cublasDgemvStridedBatched(
                            handle,
                            CUBLAS_OP_N,
                            resultshape[resultshape.size() - 1],
                            input0shape[input0shape.size() - 1],
                            &alpha,
                            weight_t.get(),
                            weight_t.shape().back(),
                            0,
                            inputget,
                            1,
                            step1,
                            &beta,
                            resultget,
                            1,
                            stepresult,
                            batch_size
                        ));
                    }


                }
                else{
                    for(int  offset1 = 0 , offsetresult = 0;offsetresult < result.size(); offset1 += step1 , offsetresult += stepresult){
                        for(int i = 0;i<result.shape()[result.shape().size() - 1];i++){
                                T sum = 0.0;
                                for(int k = 0;k<input0shape[input0shape.size() - 1];k++){
                                    sum += weight.get()[ i * input0stride[1] + k * input0stride[0]] 
                                        * input.get()[offset1 + k ];
                                }
                                sum += bias.get()[i];
                                result.get()[offsetresult + i ] = sum;
                        }
                    }
                }
                if(input.requires_grad()){
                    result.set_grad_fn(
                        std::make_shared<Functional::ModuleFunctionWrapper<T> >(this , input  ));
                    result.set_requires_grad(true);
                }
                return std::move(result);
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                auto & input = input_cache;
                Tensor<T> grad_input(input.shape() , input.device());
                Tensor<T> grad_weight = make_view(weight.get_grad() , weight.shape());
                Tensor<T> grad_bias = make_view(bias.get_grad() , bias.shape());
                auto inputstrides = input.get_strides();
                inputstrides.push_back(input.size());
                auto gradoutstrides = grad_out.get_strides();
                gradoutstrides.push_back(grad_out.size());
                int stepinput = inputstrides[1];
                int stepgradout = gradoutstrides[1];

                const T * inputget = input.get();
                const T * weighttget = weight_t.get();
                const T * biasget = bias.get();
                const T * gradoutget = grad_out.get();
                T * gradinputget = grad_input.get();
                T * gradweightget = grad_weight.get();
                T * gradbiasget = grad_bias.get();

                if(input.device() == Cuda){
                    T alpha = 1.0f;
                    T beta = 1.0f;

                    for(int inputoffset = 0 , gradoutoffset = 0;gradoutoffset < grad_out.size();inputoffset += stepinput , gradoutoffset += stepgradout){

                        cublasSetStream_v2(handle , stream_gemm);
                        
                        if constexpr (std::is_same_v<T , float>){
                            CHECK_CUBLAS(
                            cublasSger_v2(
                                handle,
                                input.shape().back(),
                                grad_out.shape().back(),
                                &alpha,
                                inputget + inputoffset,
                                1,

                                gradoutget + gradoutoffset,
                                1,
                                gradweightget,
                                grad_weight.shape().back()
                            ));
                        }
                        else if constexpr (std::is_same_v<T , double>){
                            CHECK_CUBLAS(
                            cublasDger_v2(
                                handle,
                                input.shape().back(),
                                grad_out.shape().back(),
                                &alpha,
                                inputget + inputoffset,
                                1,

                                gradoutget + gradoutoffset,
                                1,
                                gradweightget,
                                grad_weight.shape().back()
                            ));
                        }
                        cublasSetStream(handle , stream_add);

                        


                        _linear_add<T><<<CudaGetBlocks(grad_bias.size()) , kCudaThreadsNum , 0 , stream_add>>>(
                            gradbiasget , 
                            gradoutget + gradoutoffset , 
                            grad_bias.size()
                        );

                    }

                    if(optimal && input.get_grad_fn() == nullptr){
                        // no need to compute grad_input
                        return {std::move(grad_input)};
                    }

                    
                    auto batch_size = grad_out.size() / stepgradout;
                    if constexpr (std::is_same_v<T , float>){
                        CHECK_CUBLAS(
                        cublasSgemvStridedBatched(
                            handle,
                            CUBLAS_OP_N,
                            input.shape().back(),
                            grad_out.shape().back(),
                            &alpha,
                            weight.get(),
                            weight.shape().back(),
                            0,
                            gradoutget,
                            1,
                            stepgradout,
                            &beta,
                            gradinputget,
                            1,
                            stepinput,
                            grad_out.size() / stepgradout
                        ));
                    }
                    else if constexpr (std::is_same_v<T , double>){
                        CHECK_CUBLAS(
                        cublasDgemvStridedBatched(
                            handle,
                            CUBLAS_OP_N,
                            input.shape().back(),
                            grad_out.shape().back(),
                            &alpha,
                            weight.get(),
                            weight.shape().back(),
                            0,
                            gradoutget,
                            1,
                            stepgradout,
                            &beta,
                            gradinputget,
                            1,
                            stepinput,
                            grad_out.size() / stepgradout
                        ));
                    }

                }
                else{
                    int m = grad_out.shape().back();
                    int n = input.shape().back();
                    for(int inputoffset = 0 , gradoutoffset = 0;gradoutoffset < grad_out.size();inputoffset += stepinput , gradoutoffset += stepgradout){
                        for(int i = 0;i<m;i++){
                            grad_bias.get()[i] += grad_out.get()[gradoutoffset + i];
                        }
                        for(int i = 0;i< n;i++){
                            T sum = 0;
                            for(int k = 0;k< m;k++){

                                sum += weight.get()[i + k * n] * 
                                    grad_out.get()[gradoutoffset + k];
                            }
                            grad_input.get()[inputoffset + i] += sum;
                        }
                        for(int i = 0;i< m;i++){
                            for(int j = 0;j< n;j++){
                                grad_weight.get()[i * grad_weight.shape().back() + j] += 
                                    grad_out.get()[gradoutoffset + i] * 
                                    input.get()[inputoffset + j];
                            }
                        }
                    }
                }

                return {std::move(grad_input)};
            }
            std::vector<Tensor<T>> parameters() override{
                return {weight , bias};
            }
    };



    template <typename T>
    __global__ void _conv_bias_init(T * result , const T * bias , const int size , const int stride , const int out_channel){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size){
            int cid = (index / stride) % out_channel;
            result[index] = bias[cid];
        }
    }
    // kernels specific for 2d conv im2col and col2im
    template <typename T , bool trans = false>  // normal mode
    __global__ void im2col_gpu_2d(T * col , const T * im,
        const int n, const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );
    template <typename T> // non transpose mode
    __global__ void col2im_gpu_2d(T * im , T * col , const int n, // n for im
        const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );
    template <typename T , bool trans = false>
    void im2col_2d(T * col , const T * im, const int n,  // n for col
        const int channels , const int height, const int width,
        const int kh, const int kw , const int pad_h , const int pad_w , 
        const int stride_h , const int stride_w , 
        const int height_col , const int width_col , const Device device){
        if(device == Cuda){
            im2col_gpu_2d<T , trans><<<CudaGetBlocks(n) , kCudaThreadsNum>>>(
                col , im , n , channels , height ,  width , 
                kh , kw , pad_h , pad_w , 
                stride_h , stride_w , 
                height_col , width_col
            );
        }
        else{
        if constexpr(!trans){
            const int ckernelsize = kh * kw * channels; // size of whole kernel
            for( int c_im = 0; c_im < channels; c_im++){
                for(int h_col = 0; h_col < height_col; h_col++){
                    for(int w_col = 0; w_col < width_col; w_col++){
                        const int h_offset = h_col * stride_h - pad_h;
                        const int w_offset = w_col * stride_w - pad_w;
                        T * col_ptr = col + (ckernelsize * (h_col * width_col + w_col) + c_im * kh * kw);
                        const T * im_ptr = im + (c_im * height + h_offset) * width + w_offset;
                        for(int i = 0; i < kh ; i++){
                            for(int j = 0; j < kw; j++){
                                int h_im = i + h_offset;
                                int w_im = j + w_offset;
                                *col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                                    im_ptr[i * width + j] : 0;
                                
                                col_ptr ++;
                            }
                        }
                    }
                }
            }
        }
        else{
            for( int c_im = 0; c_im < channels; c_im++){
                const int c_col = c_im * kh * kw;
                for(int h_col = 0; h_col < height_col; h_col++){
                    for(int w_col = 0; w_col < width_col; w_col++){
                        const int h_offset = h_col * stride_h - pad_h;
                        const int w_offset = w_col * stride_w - pad_w;
                        T * col_ptr = col + (c_col * height_col + h_col) * width_col + w_col;
                        const T * im_ptr = im + (c_im * height + h_offset) * width + w_offset;
                        for(int i = 0; i < kh ; i++){
                            for(int j = 0; j < kw; j++){
                                int h_im = i + h_offset;
                                int w_im = j + w_offset;
                                *col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                                    im_ptr[i * width + j] : 0;

                                col_ptr += height_col * width_col;
                            }
                        }
                    }
                }
            }

        }
        }
    }

    template <typename T> // non transpose mode
    void col2im_2d(T * im , T * col , const int n, // n for im
        const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col , const Device device
    ){
        if(device == Cuda){
            col2im_gpu_2d<T><<<CudaGetBlocks(n) , kCudaThreadsNum>>>(
                im , col , n , channels , height ,  width , 
                kh , kw , pad_h , pad_w , 
                stride_h , stride_w , 
                height_col , width_col
            );
        }
        else{
            const int ckernelsize = kh * kw * channels; // size of whole kernel
            int index = 0;
            for(int c_im = 0; c_im < channels;c_im++){
                for(int h_im_ = 0 ; h_im_ < height; h_im_++){
                    for(int w_im_ = 0; w_im_ < width; w_im_++){

                        T val = 0;
                        const int w_im = w_im_ + pad_w;
                        const int h_im = h_im_ + pad_h;

                        const int w_col_start = (w_im < kw) ? 0 : (w_im - kw) /stride_w + 1;
                        const int w_col_end = min(w_im / stride_w + 1 , width_col);
                        const int h_col_start = (h_im < kh) ? 0 : (h_im - kh) /stride_h + 1;
                        const int h_col_end = min(h_im / stride_h + 1 , height_col);

                        for(int h_col = h_col_start; h_col < h_col_end; h_col++){
                            for(int w_col = w_col_start; w_col < w_col_end; w_col++){
                                int w_k = w_im - w_col * stride_w;
                                int h_k = h_im - h_col * stride_h;

                                int col_index = (h_col * width_col + w_col) * ckernelsize + (c_im * kh   +   h_k) * kw + w_k;
                                val += col[col_index];
                                // set col to 0 after use
                                col[col_index] = 0;
                            }
                        }
                        im[index] = val;
                        index++;

                    }
                }
            }
            
        }
    }



    // kernels for n dim conv im2col and col2im

    template <typename T>
    __global__ void _conv_bias_backward(
        T * grad_bias,
        const T * grad_out , const int hw , const int c , const int size){
        // c for cout
        extern __shared__ char smem[];
        double * bn_smem = reinterpret_cast<double *>(smem);
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int warpId = tid / 32;
        int laneId = tid % 32;
        constexpr int warpsPerBlock = kCudaThreadsNum / 32;
        double sum = 0;
        int chw = c * hw; // fan_out
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                sum += grad_out[i + offset];
            }
        }
        sum = warpReduceSum<T>(sum);
        if(laneId == 0) bn_smem[warpId] = sum;
        __syncthreads();
        if(tid < warpsPerBlock){
            sum = bn_smem[tid];
            sum = warpReduceSum<double>(sum);
            if(tid == 0){
                grad_bias[cid] = sum;
            }
        }
    }

    template <typename T>
    class Conv2d : public Module<T>{
        private:
            Tensor<T> input_cache;
            Device device;
            int in_channels;
            int out_channels;
            int kh;
            int kw;
            int pad_h;
            int pad_w;
            int stride_h;
            int stride_w;
            T * buf;
            int buf_size;
            cublasHandle_t handle;
        public:
            Tensor<T> kernel;
            Tensor<T> bias;
            Conv2d(const int in_channels , const int out_channels ,
                 const int kh , const int kw , const int pad_h = 0 , const int pad_w = 0 , const int stride_h = 1 , const int stride_w = 1, Device device = DefaultDevice) : 
                 device(device) , kh(kh) , kw(kw) , pad_h(pad_h) , pad_w(pad_w) , stride_h(stride_h) , stride_w(stride_w) , in_channels(in_channels) , out_channels(out_channels){
                if(kh % 2 == 0 || kw % 2 == 0){
                    std::cerr << "Conv2d: kernel size must be odd" << std::endl;
                    throw std::runtime_error("Conv2d: kernel size must be odd");
                }
                int fan_in = in_channels * kh * kw;
                T inv = nn_rsqrt<T>(fan_in);
                kernel = rand<T>({out_channels , in_channels , kh , kw} , device) * 2 * inv  - inv;
                bias = rand<T>({out_channels} , device) * 2 * inv - inv;
                buf = 0;
                buf_size = 0;
                if(device == Cuda){
                    CHECK_CUBLAS(
                        cublasCreate(&handle)
                    );
                }
            }
            ~Conv2d(){
                if(buf){
                    if(device == Cuda){
                        CHECK(cudaFree(buf));
                    }
                    else{
                        delete [] buf;
                    }
                }
                if(device == Cuda){
                    CHECK_CUBLAS(
                        cublasDestroy(handle)
                    );
                }
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                auto input = inputs[0];
                if(input.ndim() != kernel.ndim()){
                    std::cerr << "Conv2d: input and kernel must have the same number of dimensions" << std::endl;
                    throw std::runtime_error("Conv2d: input and kernel must have the same number of dimensions");
                }
                if(input.shape()[1] != in_channels){
                    std::cerr << "Conv2d: input and kernel must have the same number of input channels:"
                        << input.shape()[1] << " != " << in_channels
                     << std::endl;
                    throw std::runtime_error("Conv2d: input and kernel must have the same number of input channels");
                }
                if(input.device() != device){
                    std::cerr << "Conv2d: input and kernel must be on the same device" << std::endl;
                    throw std::runtime_error("Conv2d: input and kernel must be on the same device");
                }
                if(input.requires_grad()){
                    input_cache = input;
                }
                const int b = input.shape()[0];
                const int h = input.shape()[2];
                const int w = input.shape()[3];
                
                T alpha1 = 1.0 , beta1 = 1.0;
                int height_col = (h + 2 * pad_h -  kh  ) / stride_h + 1;
                int width_col = (w + 2 * pad_w - kw) / stride_w + 1;
                int ckernelsize = kh * kw * in_channels;

                Tensor<T> result({b , out_channels , height_col , width_col} , false , device);
                int resultcoutstep = height_col * width_col;
                int resultbatchstep = resultcoutstep * out_channels , inputbatchstep = h * w * in_channels;
                int n = in_channels * resultcoutstep;
                T * resultget = result.get();
                const T * inputget = input.get();
                T * kernelget = kernel.get();
                const int resultsize = result.size();

                if(device == Cuda){
                    if(buf_size < resultcoutstep * ckernelsize){
                        if(buf){
                            CHECK(cudaFree(buf));
                        }
                        buf_size = resultcoutstep * ckernelsize;
                        CHECK(cudaMalloc(&buf , buf_size * sizeof(T) ));
                    }
                    _conv_bias_init<T><<<CudaGetBlocks(resultsize) , kCudaThreadsNum >>>(
                        resultget , bias.get() , resultsize , resultcoutstep , out_channels
                    );
                    for(int inputbatchoffset = 0 , resultbatchoffset = 0
                        ; resultbatchoffset < resultsize
                        ; inputbatchoffset += inputbatchstep , resultbatchoffset += resultbatchstep){


                        im2col_2d<T , false>(buf , inputget + inputbatchoffset,
                            n, in_channels, h , w,
                            kh , kw , pad_h , pad_w , stride_h , stride_w,
                            height_col , width_col,device
                        );



                        

                        CHECK_CUBLAS(
                            cublasSgemvStridedBatched(
                                handle,
                                CUBLAS_OP_T,
                                ckernelsize,
                                resultcoutstep,
                                &alpha1,
                                buf,
                                ckernelsize,
                                0,
                                kernelget,
                                1,
                                ckernelsize,
                                &beta1,
                                resultget + resultbatchoffset,
                                1,
                                resultcoutstep,
                                out_channels
                            )
                        );
                    }
                }
                else{
                    if(buf_size < resultcoutstep * ckernelsize){
                        if(buf){
                            delete [] buf;
                        }
                        buf_size = resultcoutstep * ckernelsize;
                        buf = new T[buf_size];
                    }
                    for(int i = 0;i < resultsize;i++){
                        resultget[i] = bias.get()[(i / resultcoutstep) % out_channels];
                    }
                    for(int inputbatchoffset = 0 , resultbatchoffset = 0
                        ; resultbatchoffset < result.size()
                        ; inputbatchoffset += inputbatchstep , resultbatchoffset += resultbatchstep){

                        im2col_2d<T , false>(buf , inputget + inputbatchoffset,
                            n, in_channels, h , w,
                            kh , kw , pad_h , pad_w , stride_h , stride_w,
                            height_col , width_col,device
                        );

                        for(int kernelcoutoffset = 0 , resultcoutoffset = 0;
                        kernelcoutoffset < kernel.size();
                        kernelcoutoffset += ckernelsize, resultcoutoffset += resultcoutstep){
                            for(int i = 0; i < resultcoutstep;i++){
                                T sum = 0;
                                for(int k = 0;k < ckernelsize;k++){
                                    sum += buf[i * ckernelsize + k] * kernelget[kernelcoutoffset + k];
                                }
                                resultget[resultbatchoffset + resultcoutoffset + i] += sum;
                            }
                        }

                    }

                }
                if (input.requires_grad()){
                    result.set_requires_grad(true);
                    result.set_grad_fn(
                        std::make_shared<Functional::ModuleFunctionWrapper<T>>(this , input)
                    );
                }
                return std::move(result);


            }

        
        
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                auto & input = input_cache;
                const int b = input.shape()[0];
                const int h = input.shape()[2];
                const int w = input.shape()[3];
                const int ckernelsize = kh * kw * in_channels;
                const int height_col = grad_out.shape()[2];
                const int width_col = grad_out.shape()[3];
                
                T alpha1 = 1.0 , beta1 = 1.0;

                int resultcoutstep = height_col * width_col;
                int resultbatchstep = resultcoutstep * out_channels , inputbatchstep = h * w * in_channels;
                const int im2col_n = resultcoutstep * in_channels;
                const int col2im_n = h * w * in_channels;

                Tensor<T> grad_kernel = make_view(kernel.get_grad() , kernel.shape());
                Tensor<T> grad_bias = make_view(bias.get_grad() , bias.shape());
                Tensor<T> grad_in(input.shape() , device);

                const T * inputget = input.get();
                const T * gradoutget = grad_out.get();
                const T * kernelget = kernel.get();
                T * gradkernelget = grad_kernel.get();
                T * gradinget = grad_in.get();
                T * gradbiasget = grad_bias.get();

                int gradoutsize = grad_out.size();

                bool is_1x1_ = kh == 1 && kw == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0;

                if(device == Cuda){
                    if(is_1x1_){
                        // no need to im2col
                        for(int inputbatchoffset = 0 , resultbatchoffset = 0;
                                resultbatchoffset < gradoutsize;
                            inputbatchoffset += inputbatchstep , resultbatchoffset += resultbatchstep){

                            CHECK_CUBLAS(
                                cublasSgemvStridedBatched(handle,
                                    CUBLAS_OP_T,
                                    resultcoutstep,
                                    ckernelsize,
                                    &alpha1,
                                    inputget + inputbatchoffset,
                                    resultcoutstep,
                                    0,
                                    gradoutget + resultbatchoffset,
                                    1,
                                    resultcoutstep,
                                    &beta1,
                                    gradkernelget,
                                    1,
                                    ckernelsize,
                                    out_channels
                                )
                            );
                        }

                    }
                    else{
                        for(int inputbatchoffset = 0 , resultbatchoffset = 0;
                                resultbatchoffset < gradoutsize;
                            inputbatchoffset += inputbatchstep , resultbatchoffset += resultbatchstep){

                            im2col_2d<T , false>(buf , inputget + inputbatchoffset, im2col_n,
                                in_channels , h , w , 
                                kh , kw , pad_h , pad_w , 
                                stride_h , stride_w , height_col , width_col, device
                            );

                            CHECK_CUBLAS(
                                cublasSgemvStridedBatched(handle,
                                    CUBLAS_OP_N,
                                    ckernelsize,
                                    resultcoutstep,
                                    &alpha1,
                                    buf,
                                    ckernelsize,
                                    0,
                                    gradoutget + resultbatchoffset,
                                    1,
                                    resultcoutstep,
                                    &beta1,
                                    gradkernelget,
                                    1,
                                    ckernelsize,
                                    out_channels
                                )
                            );
                        }

                    }
                }
                else{

                    for(int inputbatchoffset = 0 , resultbatchoffset = 0
                        ; resultbatchoffset < gradoutsize
                        ; inputbatchoffset += inputbatchstep , resultbatchoffset += resultbatchstep){

                        im2col_2d<T , true>(buf , inputget + inputbatchoffset, im2col_n,
                            in_channels , h , w , 
                            kh , kw , pad_h , pad_w , 
                            stride_h , stride_w , height_col , width_col, device
                        );

                        for(int kernelcoutoffset = 0 , resultcoutoffset = 0;
                        kernelcoutoffset < kernel.size();
                        kernelcoutoffset += ckernelsize, resultcoutoffset += resultcoutstep){
                            for(int i = 0; i < ckernelsize;i++){
                                T sum = 0;
                                for(int k = 0;k < resultcoutstep;k++){
                                    sum += buf[i * resultcoutstep + k] * gradoutget[resultcoutoffset + k + resultbatchoffset];
                                }
                                gradkernelget[kernelcoutoffset + i] += sum;
                            }
                        }
                    }


                }

                if(device == Cuda){
                    _conv_bias_backward<<< out_channels , kCudaThreadsNum , kCudaThreadsNum / 32 * sizeof(double)>>>(
                        gradbiasget,
                        gradoutget,
                        resultcoutstep,
                        out_channels,
                        gradoutsize
                    );
                }
                else{
                    for(int cid = 0; cid < out_channels;cid++){
                        T sum_grad = 0;
                        for(int offset = cid * resultcoutstep;offset < gradoutsize; offset += resultbatchstep){
                            for(int i = 0;i < resultcoutstep;i++){
                                sum_grad += grad_out[i + offset ];
                            }
                        }
                        gradbiasget[cid] = sum_grad;

                    }
                }
                if(device == Cuda){
                    if(is_1x1_){
                        for(int inputbatchoffset = 0 , resultbatchoffset = 0;
                            resultbatchoffset < gradoutsize;
                            inputbatchoffset += inputbatchstep , resultbatchoffset += resultbatchstep){
                            for(int kernelcoutoffset = 0, resultcoutoffset = 0;
                                kernelcoutoffset < kernel.size();
                                kernelcoutoffset += ckernelsize, resultcoutoffset += resultcoutstep){
                                
                                CHECK_CUBLAS(
                                    cublasSger_v2(
                                        handle,
                                        ckernelsize,
                                        resultcoutstep,
                                        &alpha1,
                                        kernelget + kernelcoutoffset,
                                        1,
                                        gradoutget + resultbatchoffset + resultcoutoffset,
                                        1,
                                        gradinget + inputbatchoffset,
                                        ckernelsize
                                    )
                                );
                            }
                        }

                    }
                    else{
                        CHECK(cudaMemset(buf , 0 , buf_size * sizeof(T)));
                        for(int inputbatchoffset = 0 , resultbatchoffset = 0;
                            resultbatchoffset < gradoutsize;
                            inputbatchoffset += inputbatchstep , resultbatchoffset += resultbatchstep){
                            for(int kernelcoutoffset = 0, resultcoutoffset = 0;
                                kernelcoutoffset < kernel.size();
                                kernelcoutoffset += ckernelsize, resultcoutoffset += resultcoutstep){
                                
                                CHECK_CUBLAS(
                                    cublasSger_v2(
                                        handle,
                                        ckernelsize,
                                        resultcoutstep,
                                        &alpha1,
                                        kernelget + kernelcoutoffset,
                                        1,
                                        gradoutget + resultbatchoffset + resultcoutoffset,
                                        1,
                                        buf,
                                        ckernelsize
                                    )
                                );
                            }
                            col2im_2d<T>(gradinget + inputbatchoffset , buf , col2im_n,
                                in_channels , h , w , kh , kw , 
                                pad_h , pad_w , stride_h , stride_w , height_col , width_col , device);
                            

                        }

                    }



                }
                else{
                    for(int inputbatchoffset = 0 , resultbatchoffset = 0;
                        resultbatchoffset < gradoutsize;
                        inputbatchoffset += inputbatchstep , resultbatchoffset += resultbatchstep){
                        for(int kernelcoutoffset = 0, resultcoutoffset = 0;
                            kernelcoutoffset < kernel.size();
                            kernelcoutoffset += ckernelsize, resultcoutoffset += resultcoutstep){
                            
                            if(kernelcoutoffset == 0){
                                for(int i = 0; i < resultcoutstep;i++){
                                    for(int k = 0;k < ckernelsize;k++){
                                        buf[i * ckernelsize + k] = gradoutget[resultbatchoffset + resultcoutoffset + i] * kernelget[kernelcoutoffset + k];
                                    }
                                }
                            }
                            else{
                                for(int i = 0; i < resultcoutstep;i++){
                                    for(int k = 0;k < ckernelsize;k++){
                                        buf[i * ckernelsize + k] += gradoutget[resultbatchoffset + resultcoutoffset + i] * kernelget[kernelcoutoffset + k];
                                    }
                                }
                            }
                        }

                        col2im_2d<T>(gradinget + inputbatchoffset , buf , col2im_n,
                            in_channels , h , w , kh , kw , 
                            pad_h , pad_w , stride_h , stride_w , height_col , width_col , device);
                    }
                }

                return {grad_in};
            }

            std::vector<Tensor<T>> parameters() override{
                return {kernel , bias};
            }

    };



    template <typename T>
    class MaxPool2d : public Module<T>{
        private:
            std::vector<int> kernel_shape_;
            std::shared_ptr<Functional::MaxPool2dFunc<T>> pool2d_func;
        public:
            MaxPool2d(std::vector<int> kernel_shape , Device device = DefaultDevice) : kernel_shape_(kernel_shape){
                pool2d_func = std::make_shared<Functional::MaxPool2dFunc<T>>(kernel_shape_ , device);
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & input) override{
                Tensor<T> result = pool2d_func->forward(input);
                if(input[0].requires_grad()){
                    result.set_grad_fn(pool2d_func);
                }
                return result;
            }
            std::vector<Tensor<T>> parameters() override{
                return {};
            }
    };

    

    __global__ void _softmax_kernel_small_512f(float * output, const float * input, const int N, const int C);
    __global__ void _softmax_kernel_small_512d(double * output, const double * input, const int N, const int C);
    __global__ void _softmax_kernel_general_f(float * output, const float * input, const int N, const int C);
    __global__ void _softmax_kernel_general_d(double * output, const double * input, const int N, const int C);

    template <typename T>
    class Softmax : public Module<T>{
        public:
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                if(inputs.size() != 1){
                    throw std::runtime_error("Softmax input size must be 1");
                }
                auto input = inputs[0];
                Tensor<T> output(input.shape() , input.device());
                int stride = input.shape().back();
                int batchsize = input.size() / stride;
                if(output.device() == Cuda){
                    if constexpr(std::is_same_v<T , float>){
                        if(stride > 512){
                            _softmax_kernel_general_f<<<batchsize , kCudaThreadsNum , 2 * (kCudaThreadsNum / 32) * sizeof(float)>>>(output.get() , input.get() , batchsize , stride);
                        }
                        else{
                            _softmax_kernel_small_512f<<<batchsize , kCudaThreadsNum , 2 * (kCudaThreadsNum / 32) * sizeof(float)>>>(
                                output.get() , input.get() , batchsize , stride
                            );
                        }
                    }else if constexpr(std::is_same_v<T , double>){
                        if(stride > 512){
                            _softmax_kernel_general_d<<<batchsize , kCudaThreadsNum , 2 * (kCudaThreadsNum / 32) * sizeof(double)>>>(
                                output.get() , input.get() , batchsize , stride
                            );
                        }
                        else{
                            _softmax_kernel_small_512d<<<batchsize , kCudaThreadsNum , 2 * (kCudaThreadsNum / 32) * sizeof(double)>>>(
                                output.get() , input.get() , batchsize , stride
                            );
                        }

                    }
                }
                else{
                    for(int i = 0;i < batchsize;i++){
                        T maxval = -FLT_MAX;
                        for(int j = 0;j < stride;j++){
                            maxval = fmax(maxval , input.get()[i * stride + j]);
                        }
                        T sum = 0;
                        for(int j = 0;j < stride;j++){
                            T val = nn_exp<T>(input.get()[i * stride + j] - maxval);
                            output.get()[i * stride + j] = val;
                            sum += val;
                        }
                        for(int j = 0;j < stride;j++){
                            output.get()[i * stride + j] /= sum;
                        }
                    }
                }
                return output;
            }
            std::vector<Tensor<T>> parameters() override{
                return {};
            }
    };


    template <typename T>
    __global__ void _cross_entropy_backward_kernel(T * grad_input , const T * input_softmax , const T * grad_out , const T * label_cache , const int batchsize , 
    const int step){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= batchsize * step){
            return;
        }
        grad_input[index] = (input_softmax[index] - ((int)label_cache[index / step] == index % step)) * grad_out[0] / batchsize;
    }

    template <typename T>
    __global__ void _cross_entropy_forward_kernel(T * loss , const T * input , const T * label_cache , const int batchsize , const int step){

        extern __shared__ char shared_ce[];
        T * smem_ce = reinterpret_cast<T *>(shared_ce);
        int idx = blockIdx.x;
        int tid = threadIdx.x;
        int warpId = threadIdx.x / 32; 
        int laneId = threadIdx.x % 32;


        constexpr int warpsPerBlock = kCudaThreadsNum / 32;

        const T* x = input + idx * step;

        T maxval = -FLT_MAX;
        for (int i = tid; i < step; i += kCudaThreadsNum) {
            maxval = fmaxf(maxval, x[i]); 
        }
        maxval = warpReduceMax<T>(maxval);
        if (laneId == 0) 
            smem_ce[warpId] = maxval;
        __syncthreads();
        if (tid < warpsPerBlock) {
            maxval = smem_ce[tid];
            maxval = warpReduceMax<T>(maxval);
            if(tid == 0){
                smem_ce[0] = maxval;    
            }
            // store the final max in the first position
        }
        __syncthreads();
        maxval = smem_ce[0];
        T sum = 0.0f;
        for (int i = tid; i < step; i += blockDim.x) {
            sum += nn_exp_device<T>(x[i] - maxval);
        }   
        sum = warpReduceSum<T>(sum);
        if( laneId == 0 ) 
            smem_ce[warpId] = sum;
        __syncthreads();
        if (tid < warpsPerBlock) {
            sum = smem_ce[tid];
            sum = warpReduceSum<T>(sum);
            if(tid == 0){
                smem_ce[0] = sum;
            }
        }
        __syncthreads();
        sum = smem_ce[0];

        int label = (int)label_cache[idx];
        if(tid == 0){
            atomicAdd(loss , maxval - x[label] + logf(sum));
        }
    }

    template <typename T>
    __global__ void elementDivide(T * loss , const int batchsize){
        loss[0] /= batchsize;
    }


    template <typename T>
    class CrossEntropy : public Module<T>{
        private:
            Softmax<T> softmax;
            Tensor<T> input_cache , input_softmax_cache , label_cache;
        public:
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                if(inputs.size()!= 2){
                    std::cerr << "CrossEntropy input size must be 2" << std::endl;
                    throw std::runtime_error("CrossEntropy input size must be 2");
                }
                if(inputs[0].shape()[0] != inputs[1].shape()[0]){
                    std::cerr << "CrossEntropy input size must be the same" << std::endl;
                    throw std::runtime_error("CrossEntropy input size must be the same");
                }
                auto input = inputs[0];
                auto label = inputs[1];
                auto input_softmax = softmax(input);
                if(input.requires_grad()){
                    input_cache = input;
                    input_softmax_cache = input_softmax;
                    label_cache = label;
                }
                Tensor<T> loss(T(0) , {1} , input.device());
                int batchsize = input.size() / input.shape().back();
                int step = input.shape().back();
                if(input.device() == Cpu){
                    for(int i = 0;i < batchsize;i++){
                        loss.get()[0] += -log(input_softmax.get()[i * step + (int)label.get()[i]]);
                    }
                    loss.get()[0] /= batchsize;
                }
                else{
                    _cross_entropy_forward_kernel<<<batchsize , kCudaThreadsNum , sizeof(T) * kCudaThreadsNum / 32>>>(
                        loss.get() , input.get() , label.get() , batchsize , step
                    );
                    elementDivide<T><<<1 , 1>>>(loss.get() , batchsize);
                }
                if(input.requires_grad()){
                    loss.set_requires_grad(true);
                    loss.set_grad_fn(
                        std::make_shared<Functional::ModuleFunctionWrapper<T> >(this , input  ));
                }
                return loss;
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                int batchsize = input_cache.size() / input_cache.shape().back();
                int step = input_cache.shape().back();
                Tensor<T> grad_input(input_cache.shape() , input_cache.device());
                if(input_cache.device() == Cpu){
                    for(int i = 0;i<batchsize;i++){
                        for(int j = 0;j<step;j++){
                            grad_input.get()[i * step + j] = (input_softmax_cache.get()[i * step + j]) * grad_out.get()[0] / batchsize;
                        }
                        grad_input.get()[i * step + (int)label_cache.get()[i]] -= grad_out.get()[0] / batchsize;
                    }
                }
                else{
                    _cross_entropy_backward_kernel<T><<<CudaGetBlocks(input_cache.size()) , kCudaThreadsNum>>>(
                        grad_input.get() , input_softmax_cache.get() , grad_out.get() , label_cache.get() , batchsize , step
                    );
                }
                return {std::move(grad_input)};
            }
                std::vector<Tensor<T>> parameters() override{
                    return {};
                }


    };
    template <typename T>
    class ReLU : public Module<T>{
        private:
            Tensor<T> input_cache;
            std::shared_ptr<Functional::ReLUFunc<T>> relu_func;
        public:
            ReLU() : relu_func(std::make_shared<Functional::ReLUFunc<T>>()){}
            Tensor<T> forward(const std::vector<Tensor<T>> & input) override{
                if(input[0].requires_grad()){
                    Tensor<T> result = relu_func->forward(input);
                    result.set_grad_fn(relu_func);
                    input_cache = input[0];
                    return result;
                }
                return Functional::ReLUFunc<T>().forward(input);
            }
            std::vector<Tensor<T>> parameters() override{
                return {};
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                return relu_func->backward(grad_out);
            }
    };
    template <typename T>
    class Sigmoid : public Module<T>{
        private:
            Tensor<T> input_cache;
            std::shared_ptr<Functional::SigmoidFunc<T>> sigmoid_func;
        public:
            Tensor<T> forward(const std::vector<Tensor<T>> & input) override{
                if(input[0].requires_grad()){
                    Tensor<T> result = sigmoid_func->forward(input);
                    result.set_grad_fn(sigmoid_func);
                    input_cache = input[0];
                    return result;
                }
                return Functional::SigmoidFunc<T>().forward(input);
            }
            std::vector<Tensor<T>> parameters() override{
                return {};
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                return sigmoid_func->backward(grad_out);
            }
    };

    template <typename T>
    __global__ void _batch_norm_forward_kernel(
        T * result , T * xhat_cache, T * var_inv_cache, T * running_mean , T * running_var,
        const T * input , const T * gamma , const T * beta , const int hw , const int c , const int batch_size , const int size ,const T momentum){
        extern __shared__ char smem[];
        T * bn_smem = reinterpret_cast<T *>(smem);
        constexpr T epsilon = 1e-5;
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int warpId = tid / 32;
        int laneId = tid % 32;
        constexpr int warpsPerBlock = kCudaThreadsNum / 32;
        T sum = 0;
        int chw = c * hw;
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                sum += input[i + offset];
            }
        }
        sum = warpReduceSum<T>(sum);
        T hw_batch_size = hw * batch_size;
        __syncthreads();
        if(laneId == 0) bn_smem[warpId] = sum / hw_batch_size;
        __syncthreads();
        if(tid < warpsPerBlock){
            sum = bn_smem[tid];
            sum = warpReduceSum<T>(sum);
            if(tid == 0){
                running_mean[cid] = sum * (1 - momentum) + running_mean[cid] * momentum;
                bn_smem[0] = sum;
            }
        }
        __syncthreads();
        sum = 0;
        const T mean = bn_smem[0];
        
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                T diff = input[i + offset] - mean;
                sum += diff * diff;
            }
        }
        sum = warpReduceSum<T>(sum);
        __syncthreads();
        if(laneId == 0) bn_smem[warpId] = sum / hw_batch_size;
        __syncthreads();
        if(tid < warpsPerBlock){
            sum = bn_smem[tid];
            sum = warpReduceSum<T>(sum);
            if(tid == 0){
                var_inv_cache[cid] = nn_rsqrt<T>(sum + epsilon);
                running_var[cid] = sum * (1 - momentum) + running_var[cid] * momentum;
            }
        }
        __syncthreads();
        T var_inv = var_inv_cache[cid];
        T gamma_cid = gamma[cid];
        T beta_cid = beta[cid];
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                T diff = input[i + offset] - mean;
                T xhat = diff * var_inv;
                xhat_cache[i + offset] = xhat;
                result[i + offset] = gamma_cid * xhat + beta_cid;
            }
        }
    }
    template <typename T>
    __global__ void _batch_norm_kernel2(
        T * output , const T * input , const T * running_mean , const T * var_inv_cache , const T * gamma , const T * beta , const T epsilon,
        const int hw , const int c , const int batch_size , const int size
    ){
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int chw = c * hw;
        T mean = running_mean[cid];
        T var_inv = var_inv_cache[cid];
        T gamma_cid = gamma[cid];
        T beta_cid = beta[cid];
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                T diff = input[i + offset] - mean;
                output[i + offset] = gamma_cid * diff * var_inv + beta_cid;
            }
        }
    }

    template <typename T>
    __global__ void _batch_norm_kernel_gamma_beta(
        T * grad_gamma , T * grad_beta , const T * grad_out , const T * input , const T * xhat_cache  , const T epsilon,
        const int hw , const int c  , const int size
    ){
        extern __shared__ char smem[];
        T * bn_smem = reinterpret_cast<T *>(smem);
        constexpr int warpsPerBlock = kCudaThreadsNum / 32;
        T * bn_smem_beta = bn_smem + warpsPerBlock;
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int warpId = tid / 32;
        int laneId = tid % 32;
        T sum_gamma = 0;
        T sum_beta = 0;
        int chw = c * hw;
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                sum_gamma += grad_out[i + offset] * xhat_cache[i + offset];
                sum_beta += grad_out[i + offset];
            }
        }
        sum_gamma = warpReduceSum<T>(sum_gamma);
        sum_beta = warpReduceSum<T>(sum_beta);
        __syncthreads();
        if(laneId == 0){ 
             bn_smem[warpId] = sum_gamma;
             bn_smem_beta[warpId] = sum_beta;
        }
        __syncthreads();
        if(tid < warpsPerBlock){
            sum_gamma = bn_smem[tid];
            sum_beta = bn_smem_beta[tid];
            sum_gamma = warpReduceSum<T>(sum_gamma);
            sum_beta = warpReduceSum<T>(sum_beta);
            if(tid == 0){
                grad_gamma[cid] = sum_gamma;
                grad_beta[cid] = sum_beta;
            }
        }
    }

    template <typename T>
    __global__ void _batch_norm_kernel_in(
        T * grad_in , const T * grad_out , const T * xhat_cache , const T * var_inv_cache , const T * grad_beta , const T * grad_gamma , const T * gamma , const T epsilon,
        const int hw , const int c , const int batch_size , const int size 
    ){
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int chw = c * hw;
        int hw_batch_size = hw * batch_size;
        T var_inv = var_inv_cache[cid];
        T gamma_cid = gamma[cid];
        T coefficient = var_inv * gamma_cid / hw_batch_size;
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                grad_in[i + offset ] = (hw_batch_size * grad_out[i + offset] - grad_beta[cid] - grad_gamma[cid] * xhat_cache[i + offset]) * coefficient;
            }
        }
    }

    template <typename T>
    __global__ void _calculate_inv_var(T * var_inv_cache , const T * var ,const int size , const T epsilon){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < size){
            var_inv_cache[index] = nn_rsqrt<T>(var[index] + epsilon);
        }
    }



    template <typename T>
    class BatchNorm2d : public Module<T>{
        Tensor<T> xhat_cache , var_inv_cache;
        Tensor<T> input_cache;
        public:
        Tensor<T> gamma , beta;
        Tensor<T> running_mean , running_var;
        T momentum;
        T epsilon;
        int c;
            BatchNorm2d(const int num_features , T momentum = 0.1 , Device device = DefaultDevice) : c(num_features) , momentum(momentum){
                gamma = ones<T>({num_features} , device);
                beta = zeros<T>({num_features} , device);
                running_mean = zeros<T>({num_features} , device);
                running_var = ones<T>({num_features} , device);
                var_inv_cache = Tensor<T>({num_features} , false ,  device);
                epsilon = 1e-5;
            }
            void train() override{
                this->training = true;
            }
            void eval() override{
                this->training = false;
                _calculate_inv_var<T><<< CudaGetBlocks(c) , kCudaThreadsNum>>>(var_inv_cache.get() , running_var.get() , c , epsilon);
                xhat_cache = Tensor<T>(); // clear cache
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                if(inputs.size() != 1){
                    std::cerr << "BatchNorm forward input size must be 1" << std::endl;
                    throw std::runtime_error("BatchNorm forward input size must be 1");
                }
                auto input = inputs[0];
                if(input.requires_grad() && this->training){
                    input_cache = input;
                }
                auto inputshape = input.shape();
                Tensor<T> result(inputshape , 0 , input.device());
                if(xhat_cache.is_null() || xhat_cache.shape() != inputshape){
                    xhat_cache = Tensor<T>(inputshape , false , input.device());
                }
                if(this->training){
                    int n = inputshape[0];
                    int h = inputshape[2];
                    int w = inputshape[3];
                    int hw = h * w;
                    int chw = c * hw;
                    int size = chw * n;
                    int hw_batch_size = hw * n;
                    if(input.device() == Cuda){
                        _batch_norm_forward_kernel<T><<<c , kCudaThreadsNum , sizeof(T) * (kCudaThreadsNum / 32)>>>(
                            result.get(),
                            xhat_cache.get() , var_inv_cache.get() , running_mean.get() , running_var.get() , input.get() , gamma.get() , beta.get() , hw  , c , n , size , momentum
                        );
                    }
                    else{
                        for(int cid = 0; cid < c;cid++){
                            T mean = 0;
                            T var = 0;
                            for(int offset = cid * hw;offset < size; offset += chw){
                                for(int i = 0;i < hw;i++){
                                    mean += input[i + offset];
                                }
                            }
                            mean /= hw_batch_size;

                            for(int offset = cid * hw;offset < size; offset += chw){
                                for(int i = 0;i < hw;i++){
                                    var += (input[i + offset] - mean) * (input[i + offset] - mean);
                                }
                            }
                            var /= hw_batch_size;
                            T var_inv = nn_rsqrt<T>(var + epsilon);
                            var_inv_cache[cid] = var_inv;
                            running_mean[cid] = mean * (1 - momentum) + running_mean[cid] * momentum;
                            running_var[cid] = var * (1 - momentum) + running_var[cid] * momentum;
                            T gamma_cid = gamma[cid];
                            T beta_cid = beta[cid];
                            for(int offset = cid * hw;offset < size; offset += chw){
                                for(int i = 0;i < hw;i++){
                                    T diff = input[i + offset] - mean;
                                    xhat_cache[i + offset] = diff * var_inv;
                                    result[i + offset] = gamma_cid * xhat_cache[i + offset] + beta_cid;
                                }
                            }
                        }
                    }
                }
                else{
                    int n = inputshape[0];
                    int c = inputshape[1];
                    int h = inputshape[2];
                    int w = inputshape[3];
                    int hw = h * w;
                    int chw = c * hw;
                    int size = chw * n;
                    if(input.device() == Cuda){
                        _batch_norm_kernel2<T><<<c , kCudaThreadsNum>>>(
                            result.get() , input.get() , running_mean.get() , var_inv_cache.get() , gamma.get() , beta.get() , epsilon,
                            hw  , c , n , size
                        );
                    }
                    else{
                        for(int cid = 0; cid < c;cid++){
                            T var_inv = var_inv_cache[cid];
                            T mean = running_mean[cid];
                            T gamma_cid = gamma[cid];
                            T beta_cid = beta[cid];
                            for(int offset = cid * hw;offset < size; offset += chw){
                                for(int i = 0;i < hw;i++){
                                    T diff = input[i + offset] - mean;
                                    result[i + offset] = gamma_cid * diff * var_inv + beta_cid;
                                }
                            }
                        }
                    }
                }
                if (input.requires_grad()){
                    result.set_requires_grad(true);
                    result.set_grad_fn(
                        std::make_shared<Functional::ModuleFunctionWrapper<T> >(this , input  ));
                }
                return std::move(result);
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                auto input = input_cache;
                auto inputshape = grad_out.shape();
                Tensor<T> grad_in = Tensor<T>(inputshape , input.device());
                int n = inputshape[0];
                int h = inputshape[2];
                int w = inputshape[3];
                beta.alloc_grad();
                Tensor<T> grad_beta = make_view(beta.get_grad() , {c});
                gamma.alloc_grad();
                Tensor<T> grad_gamma = make_view(gamma.get_grad() , {c});
                int hw = h * w;
                int chw = c * hw;
                int hw_batch_size = hw * n;
                int size = chw * n;
                if(input.device() == Cuda){
                    _batch_norm_kernel_gamma_beta<T><<<c , kCudaThreadsNum , sizeof(T) * (kCudaThreadsNum / 32) * 2>>>(
                        grad_gamma.get() , grad_beta.get() , grad_out.get() , input.get() ,xhat_cache.get()  , epsilon,
                        hw , c , size
                    );
                    _batch_norm_kernel_in<T> <<< c , kCudaThreadsNum>>>(
                        grad_in.get() , grad_out.get() , xhat_cache.get() , var_inv_cache.get() , grad_beta.get() , grad_gamma.get() , gamma.get() , epsilon,
                        hw , c , n , size
                    );
                }
                else{
                    for(int cid = 0; cid < c;cid++){
                        T sum_grad = 0;
                        T sum_grad_gamma = 0;
                        T var_inv = var_inv_cache[cid];
                        for(int offset = cid * hw;offset < size; offset += chw){
                            for(int i = 0;i < hw;i++){
                                sum_grad += grad_out[i + offset];
                                sum_grad_gamma += grad_out[i + offset] * xhat_cache[i + offset];
                            }
                        }
                        grad_beta[cid] = sum_grad;
                        grad_gamma[cid] = sum_grad_gamma;

                        T coefficient = var_inv * gamma[cid] / hw_batch_size;
                        for(int offset = cid * hw;offset < size; offset += chw){
                            for(int i = 0;i < hw;i++){
                                grad_in[i + offset] = (grad_out[i + offset] * hw_batch_size - sum_grad - sum_grad_gamma * xhat_cache[i + offset]) * coefficient;
                            }
                        }

                    }
                }
                return {std::move(grad_in)};
            }
            std::vector<Tensor<T>> parameters() override{
                return {gamma , beta};
            }

    };

    template <typename T>
    class Flatten : public Module<T>{
        int start_dim;
        int end_dim;
        public:
            Flatten(int start_dim = 0 , int end_dim = -1) : start_dim(start_dim) , end_dim(end_dim){ }
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                auto input = inputs[0];
                auto shape = input.shape();
                int newsize = 1;
                if(end_dim == -1){
                    end_dim = shape.size() - 1;
                }
                for(int i = start_dim;i <= end_dim;i++){
                    newsize *= shape[i];
                }
                std::vector<int> newshape(shape.begin() , shape.begin() + start_dim);
                newshape.push_back(newsize);
                newshape.insert(newshape.end() , shape.begin() + end_dim + 1 , shape.end());
                return input.reshape(newshape);
            }
    };

    template <typename T>
    class ResBlock : public Module<T>{
        std::shared_ptr<Sequential<T>> left , shortcut;
        std::vector<Tensor<T>> params;
        public:
            ResBlock(const int in_channels , const int out_channels , const int stride = 1 , const Device device = DefaultDevice){
                left = std::make_shared<Sequential<T>>(std::vector<std::shared_ptr<Module<T>>>({
                    std::make_shared<Conv2d<T>>(in_channels , out_channels , 3 , 3  , 1 , 1 , stride , stride , device),
                    std::make_shared<BatchNorm2d<T>>(out_channels, 0.1 , device),
                    std::make_shared<ReLU<T>>(),
                    std::make_shared<Conv2d<T>>(out_channels , out_channels , 3 , 3  , 1 , 1 , 1 , 1 , device),
                    std::make_shared<BatchNorm2d<T>>(out_channels, 0.1 , device)
                }));

                shortcut = std::make_shared<Sequential<T>>(std::vector<std::shared_ptr<Module<T>>>({}));

                if( stride != 1 || in_channels != out_channels){
                    shortcut = std::make_shared<Sequential<T>>(std::vector<std::shared_ptr<Module<T>>>({
                        std::make_shared<Conv2d<T>>(in_channels , out_channels , 1 , 1 , 0 , 0 , stride , stride , device),
                        std::make_shared<BatchNorm2d<T>>(out_channels, 0.1 , device)
                    }));
                }
                params = left->parameters();
                auto shortparams = shortcut->parameters();
                params.insert(params.end() , shortparams.begin() , shortparams.end());
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                auto out = left->forward(inputs);
                out = out +  shortcut->forward(inputs);
                out = out.relu();
                return out;
            }
            std::vector<Tensor<T>> parameters() override{
                return params;
            }
            void train() override{
                left->train();
                shortcut->train();
            }
            void eval() override{
                left->eval();
                shortcut->eval();
            }
    };
    template <typename T>
    class ResNet18 : public Module<T>{
        public:
        std::shared_ptr<Sequential<T>> conv1;
        std::shared_ptr<Sequential<T>> layer1 , layer2 , layer3 , layer4;
        std::shared_ptr<Linear<T>> fc;
        std::vector<Tensor<T>> params;
        int in_channels , h , w;
        cudaStream_t streams[kStreamCount];
        ResNet18(const int num_classes , const int h = 32 , const int w = 32) : h(h) , w(w){
            in_channels = 64;
            conv1 = std::make_shared<Sequential<T>>(std::vector<std::shared_ptr<Module<T>>>({
                std::make_shared<Conv2d<T>>(3 , 64 , 3 , 3 , 1 , 1, 1 , 1),
                std::make_shared<BatchNorm2d<T>>(64),
                std::make_shared<ReLU<T>>()
            }));
            layer1 = make_layer(64 , 2 , 1);
            layer2 = make_layer(128 , 2 , 2);
            layer3 = make_layer(256 , 2 , 2);
            layer4 = make_layer(512 , 2 , 2);
            fc = std::make_shared<Linear<T>>(512 , num_classes);
            params = conv1->parameters();
            auto params1 = layer1->parameters();
            params.insert(params.end() , params1.begin() , params1.end());
            auto params2 = layer2->parameters();
            params.insert(params.end() , params2.begin() , params2.end());
            auto params3 = layer3->parameters();
            params.insert(params.end() , params3.begin() , params3.end());
            auto params4 = layer4->parameters();
            params.insert(params.end() , params4.begin() , params4.end());
            auto params5 = fc->parameters();
            params.insert(params.end() , params5.begin() , params5.end());
            for(int i = 0;i < kStreamCount;i++){
                CHECK(cudaStreamCreate(&streams[i]));
            }
        }
        ~ResNet18(){
            for(int i = 0;i < kStreamCount;i++){
                CHECK(cudaStreamDestroy(streams[i]));
            }
        }

        std::shared_ptr<Sequential<T>> make_layer( const int channels , const int num_blocks , const int stride){
            std::vector<int> strides = {stride};
            for(int i = 1;i < num_blocks;i++) strides.push_back(1);
            std::vector<std::shared_ptr<Module<T>>> layers;
            for(int i = 0;i < num_blocks;i++){
                layers.push_back(std::make_shared<ResBlock<T>>(in_channels , channels , strides[i]));
                in_channels = channels;
            }
            return std::make_shared<Sequential<T>>(layers);
        }

        Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
            auto out = conv1->forward(inputs);
            out = layer1->forward({out});
            out = layer2->forward({out});
            out = layer3->forward({out});
            out = layer4->forward({out});
            out = out.maxpool2d({h / 8 , w / 8});
            out = out.reshape({out.shape()[0] , 512});
            out = fc->forward({out});
            return out;
        }

        std::vector<Tensor<T>> parameters() override{
            return params;
        }

        void zero_grad() override{
            int i = 0;
            for(auto & param : params){
                param.zero_grad(streams[i % kStreamCount]);
                i++;
            }
            for(int i = 0;i < kStreamCount;i++){
                CHECK(cudaStreamSynchronize(streams[i]));
            }
        }
        void train() override{
            conv1->train();
            layer1->train();
            layer2->train();
            layer3->train();
            layer4->train();
            fc->train();
        }
        void eval() override{
            conv1->eval();
            layer1->eval();
            layer2->eval();
            layer3->eval();
            layer4->eval();
            fc->eval();
        }



    };

    template <typename T>
    class MiniResNet : public Module<T>{
        public:
        std::shared_ptr<Sequential<T>> conv1;
        std::shared_ptr<Sequential<T>> layer1 , layer2;
        std::shared_ptr<Linear<T>> fc;
        std::vector<Tensor<T>> params;
        int in_channels , h , w;
        cudaStream_t streams[kStreamCount];
        MiniResNet(const int num_classes , const int h = 32 , const int w = 32) : h(h) , w(w){
            in_channels = 64;
            conv1 = std::make_shared<Sequential<T>>(std::vector<std::shared_ptr<Module<T>>>({
                std::make_shared<Conv2d<T>>(3 , 64 , 3 , 3 , 1 , 1, 1 , 1),
                std::make_shared<BatchNorm2d<T>>(64),
                std::make_shared<ReLU<T>>()
            }));
            layer1 = make_layer(64 , 2 , 2);
            layer2 = make_layer(256 , 2 , 2);
            fc = std::make_shared<Linear<T>>(256 * h / 16 * w / 16 , num_classes);
            params = conv1->parameters();
            auto params1 = layer1->parameters();
            params.insert(params.end() , params1.begin() , params1.end());
            auto params2 = layer2->parameters();
            params.insert(params.end() , params2.begin() , params2.end());
            auto params5 = fc->parameters();
            params.insert(params.end() , params5.begin() , params5.end());
            for(int i = 0;i < kStreamCount;i++){
                CHECK(cudaStreamCreate(&streams[i]));
            }
        }
        ~MiniResNet(){
            for(int i = 0;i < kStreamCount;i++){
                CHECK(cudaStreamDestroy(streams[i]));
            }
        }

        std::shared_ptr<Sequential<T>> make_layer( const int channels , const int num_blocks , const int stride){
            std::vector<int> strides = {stride};
            for(int i = 1;i < num_blocks;i++) strides.push_back(1);
            std::vector<std::shared_ptr<Module<T>>> layers;
            for(int i = 0;i < num_blocks;i++){
                layers.push_back(std::make_shared<ResBlock<T>>(in_channels , channels , strides[i]));
                in_channels = channels;
            }
            return std::make_shared<Sequential<T>>(layers);
        }

        Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
            auto out = conv1->forward(inputs);
            out = layer1->forward({out});
            out = layer2->forward({out});
            out = out.maxpool2d({4 , 4});
            out = out.reshape({out.shape()[0] , 256 * h / 16 * w / 16});
            out = fc->forward({out});
            return out;
        }

        std::vector<Tensor<T>> parameters() override{
            return params;
        }

        void zero_grad() override{
            int i = 0;
            for(auto & param : params){
                param.zero_grad(streams[i % kStreamCount]);
                i++;
            }
            for(int i = 0;i < kStreamCount;i++){
                CHECK(cudaStreamSynchronize(streams[i]));
            }
        }
        void train() override{
            conv1->train();
            layer1->train();
            layer2->train();
            fc->train();
        }
        void eval() override{
            conv1->eval();
            layer1->eval();
            layer2->eval();
            fc->eval();
        }
        


    };

}
}


#endif