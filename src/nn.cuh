#ifndef _NN_H_
#define _NN_H_

#include "tensor.cuh"
#include <cublas_v2.h>
#include <cmath>

namespace mytorch{
namespace nn{
    constexpr size_t kCudaTransposeTileSize = 4;
    constexpr size_t kCudaMultiDimMax = 16;
    constexpr int kStreamCount = 8;
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
            virtual void set_train(bool train){
                training = train;
            }
    };

    template <typename T>
    class Sequential : public Module<T>{
        private:
            std::vector<std::shared_ptr<Module<T>>> modules_;
        public:
            Sequential(const std::vector<std::shared_ptr<Module<T>>> & modules){
                modules_ = modules;
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & input){
                Tensor<T> output = input[0];
                for(auto & module : modules_){
                    output = module->forward({output});
                }
                return output;
            }
            std::vector<Tensor<T>> parameters() override{
                std::vector<Tensor<T>> params;
                for(auto & module : modules_){
                    auto module_params = module->parameters();
                    params.insert(params.end() , module_params.begin() , module_params.end());
                }
                return params;
            }
            void set_train(bool train){
                training = train;
                for(auto & module : modules_){
                    module->set_train(train);
                }
            }
    };

    
    class CudaMultiDimIndex{
        private:
            size_t ndim_;
            size_t index_[kCudaMultiDimMax];
            size_t shape_[kCudaMultiDimMax];
        public:
            __device__ CudaMultiDimIndex(const size_t * shape ,const size_t ndim){
                ndim_ = ndim;
                for(int i = 0;i<ndim_;i++){
                    shape_[i] = shape[i];
                    index_[i] = 0;
                }
            }
            __device__ size_t * get_index(){
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
            __device__ size_t calculate_offset(const size_t * strides) const{
                size_t offset = 0;
                for(int i = 0;i<ndim_;i++){
                    offset += index_[i] * strides[ndim_ - 1 - i];
                }
                return offset;
            }
            __device__ size_t operator[](size_t i) const{
                return index_[i];
            }
    };


    namespace Functional{


        template <typename T>
        __global__ void _neg_forward_kernel(T * output , const T * input , const size_t size){
            size_t index = threadIdx.x + blockIdx.x * blockDim.x;
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
                        for(size_t i = 0;i < result.size();i++){
                            result.get()[i] = - inputs[0].get()[i];
                        }
                    }
                    result.set_requires_grad(inputs[0].requires_grad());
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T>& grad_output){
                    Tensor<T> gradin(grad_output.shape() , grad_output.device());
                    if(gradin.device() == Cuda){
                        _neg_forward_kernel<<<CudaGetBlocks(gradin.size()) , kCudaThreadsNum>>>(gradin.get() , grad_output.get() , gradin.size());
                    }
                    else{
                        for(size_t i = 0; i < gradin.size();i++){
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
        __global__ void _add_forward_kernel(T * output, const T* input1, const T* input2 ,  size_t size){
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
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    return {grad_out.deepcopy() , grad_out.deepcopy()};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a , b};
                }
        };
        template <typename T>
        __global__ void _sub_forward_kernel(T * output, const T* input1, const T* input2 ,  size_t size){
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
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_output) override{
                    return {grad_output.deepcopy() , (-grad_output).deepcopy()};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a , b};
                }
        };

        template <typename T>
        __global__ void _mul_forward_kernel(T * output, const T* input1, const T* input2 ,  size_t size){
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
                    return result;
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
        __global__ void _div_forward_kernel(T * output , const T * input1 , const T * input2 , size_t size){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size){
                output[index] = input1[index] / input2[index];
            }
        }
        template<typename T>
        __global__ void _div_backward_kernel_1(T * output , const T * grad_out , const T * input , size_t size){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index < size){
                output[index] = grad_out[index] / input[index];
            }
        }
        template <typename T>
        __global__ void _div_backward_kernel_2(T * output , const T * grad_out , const T * a , const T * b , size_t size){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
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
                    return result;
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
        __global__ void _relu_forward_kernel(T * output , const T * input , size_t size){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < size){
                output[index] = input[index] > 0 ? input[index] : 0;
            }
        }

        template<typename T>
        __global__ void _relu_forward_kernel_mask(bool * output , const T * input , size_t size){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < size){
                output[index] = input[index] > 0 ;
            }
        }

        template<typename T>
        __global__ void _relu_backward_kernel(T * grad_in , const T * grad_out , const bool * mask , size_t size){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
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
                    mask = cuda_shared_pointer<bool>(input[0].size() , input[0].device());
                    if( input[0].device() == Cuda){
                        _relu_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.get() , input[0].get() , result.size());
                        _relu_forward_kernel_mask<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(mask.get() , input[0].get() , result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.get()[i] = input[0].get()[i] > 0 ? input[0].get()[i] : 0;
                            mask.get()[i] = input[0].get()[i] > 0;
                        }
                    }
                    result.set_requires_grad(input[0].requires_grad());
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    Tensor<T> grad_input(grad_out.shape() , grad_out.device());
                    if(grad_out.device() == Cuda){
                        _relu_backward_kernel<<<CudaGetBlocks(grad_input.size()) , kCudaThreadsNum>>>(grad_input.get() , grad_out.get() , mask.get() , grad_input.size());
                    }
                    else{
                        for (int i = 0; i < grad_input.size(); i++){
                            grad_input.get()[i] = mask.get()[i] ? grad_out.get()[i] : 0;
                        }
                    }
                    return {grad_input};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a};
                }

        };

        template <typename T>
        __global__ void _sigmoid_forward_kernel(T * output , const T * input , size_t size){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < size){
                output[index] = 1.0 / (1.0 + std::expf(-input[index]));
            }
        }
        __global__ void _sum_forward_kernel_f(float * output , const float * input ,  const size_t reduce , const size_t inner);
        __global__ void _sum_forward_kernel_d(double * output , const double * input ,  const size_t reduce , const size_t inner);
        template <typename T>
        __global__ void _sum_backward_kernel(T * grad_in , const T * grad_out , const size_t reduce , const size_t inner){
            size_t ridx = threadIdx.x;
            size_t iidx = blockIdx.x % inner;
            size_t oidx = blockIdx.x / inner;
            grad_in[oidx * reduce * inner + ridx * inner + iidx] = grad_out[oidx * inner + iidx];
        }
        template <typename T>
        class SumFunc : public Function<T>{
            private :
                Tensor<T> a;
                size_t axis;
            public:
                SumFunc( const size_t axis) : axis(axis){}
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if(inputs.size() != 1){
                        throw std::runtime_error("SumFunc error!");
                    }
                    if(inputs[0].requires_grad()){
                        a = inputs[0];
                    }
                    auto input = inputs[0];
                    axis = input.ndim() - axis - 1;
                    std::vector<size_t> resultshape;
                    auto inputshape = input.shape();
                    for (int i = 0; i < inputshape.size(); i++){
                        if(i != axis){
                            resultshape.push_back(inputshape[i]);
                        }
                    }
                    Tensor<T> result(resultshape , input.device());
                    size_t indim = input.ndim();
                    size_t reduce = inputshape[axis];
                    size_t inner = 1;
                    for(size_t i = axis + 1;i < indim;i++){
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
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    auto inputshape = a.shape();
                    size_t indim = inputshape.size();
                    Tensor<T> grad_in(inputshape , a.device());
                    size_t reduce = inputshape[axis];
                    size_t inner = 1;
                    if(a.device() == Cuda){
                        for(size_t i = axis + 1;i < indim;i++){
                            inner *= inputshape[i];
                        }
                        _sum_backward_kernel<<< grad_out.size() , reduce , reduce * sizeof(T)>>>(grad_in.get() , grad_out.get() , reduce , inner);
                    }
                    else{
                        size_t outer = grad_out.size() / inner;
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
                            result.get()[i] = 1 / (1 + std::exp(-input[0].get()[i]));
                        }
                    }
                    output = result.deepcopy();
                    result.set_requires_grad(input[0].requires_grad());
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    return {(grad_out * (output * (mytorch::ones<T>(output.shape() , output.device()) - output))).deepcopy()};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a};
                }

        };

        template <typename T>
        std::vector<T> _get_transpose_vec(const std::vector<T> & input , const std::vector<size_t> & perm){
            std::vector<T> result(input.size());
            for(int i = 0;i < input.size();i++){
                result[i] = input[perm[i]];
            }
            return result;
        }
        template <typename T>
        std::vector<T> _get_transpose_vec_rev(const std::vector<T> & input , const std::vector<size_t> & perm){
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
        __global__ void _transpose_forward_kernel(T * result ,const T *  input ,const size_t size ,const size_t ndim ,const size_t * inshape 
            ,const size_t * instrides
            ,const  size_t * outstrides,
            const size_t * perm,const size_t *  revperm){
                extern __shared__ char smem[];
                T * tilem = reinterpret_cast<T *>(smem);
                size_t threadidx = threadIdx.x , blockidx = blockIdx.x;
                size_t idx[ kCudaMultiDimMax] , tileidx[ kCudaMultiDimMax];
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
                size_t index = 0;
                for(int i = 0;i< ndim;i++){
                    index += (idx[i] + kCudaTransposeTileSize * tileidx[i]) * instrides[i];
                }
                if (isvalid){
                    tilem[threadIdx.x] = input[index];
                }
                __syncthreads();
                size_t outputindex = 0;
                size_t outtileindex = 0;
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
        __global__ void _transpose_forward_kernel_2dim(T * result ,const T *  input  , const size_t m, const size_t n){
            extern __shared__ char smem[];
            T * tilem = reinterpret_cast<T *>(smem);
            size_t idx[2];
            idx[0] = threadIdx.x + kCudaTransposeTileSize * blockIdx.x; // n
            idx[1] = threadIdx.y + kCudaTransposeTileSize * blockIdx.y; // m
            size_t index = idx[0] + idx[1] * n;
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
        __global__ void _transpose_forward_kernel_HWC2CHW(T * result, const T * input , const size_t h , const size_t w , const size_t c){
            extern __shared__ char smem[];
            T * tilem = reinterpret_cast<T *>(smem);
            size_t idx[3];
            idx[0] = threadIdx.x + kCudaTransposeTileSize * blockIdx.x; // c
            idx[1] = threadIdx.y + kCudaTransposeTileSize * blockIdx.y; // w
            idx[2] = threadIdx.z + kCudaTransposeTileSize * blockIdx.z; // h
            size_t index = idx[0] + idx[1] * c + idx[2] * w * c;
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
        __global__ void _transpose_forward_kernel_CHW2HWC(T * result, const T * input , const size_t c , const size_t h , const size_t w){
            extern __shared__ char smem[];
            T * tilem = reinterpret_cast<T *>(smem);
            size_t idx[3];
            idx[0] = threadIdx.x + kCudaTransposeTileSize * blockIdx.x; // w
            idx[1] = threadIdx.y + kCudaTransposeTileSize * blockIdx.y; // h
            idx[2] = threadIdx.z + kCudaTransposeTileSize * blockIdx.z; // c
            size_t index = idx[0] + idx[1] * w + idx[2] * w * h;
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
                std::vector<size_t> perm;
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
                const int streamCount = 8;
                ~TransposeFunc(){
                    for(int i = 0;i<streamCount;i++){
                        CHECK(cudaStreamDestroy(streams[i]));
                    }
                }
                
                TransposeFunc(const std::vector<size_t> & perm) : perm(perm){
                    for(int i = 0;i<streamCount;i++){
                        CHECK(cudaStreamCreate(&streams[i]));
                    }
                    size_t eq_count = 0;
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
                    std::vector<size_t> newshape = _get_transpose_vec(input[0].shape() , perm);
                    Tensor<T> result(newshape , input[0].device());
                    result.set_requires_grad(input[0].requires_grad());
                    if(result.device() == Cuda){
                        if(ttype == TLast2Dim){


                            size_t ndim = input[0].shape().size();
                            size_t m = input[0].shape()[ndim - 2];
                            size_t n = input[0].shape()[ndim - 1];
                            size_t batchstep = m * n;
                            size_t batch_size = input[0].size()/ batchstep;

                            dim3 grid_size(divroundup(n , kCudaTransposeTileSize) , divroundup(m , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize);


                            for(size_t b = 0;b < batch_size;b ++){
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


                            size_t ndim = input[0].shape().size();
                            size_t h = input[0].shape()[ndim - 3];
                            size_t w = input[0].shape()[ndim - 2];
                            size_t c = input[0].shape()[ndim - 1];
                            size_t batchstep = h * w * c;
                            size_t batch_size = input[0].size()/ batchstep;

                            dim3 grid_size(divroundup(c , kCudaTransposeTileSize) , divroundup(w , kCudaTransposeTileSize) , divroundup(h , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize , kCudaTransposeTileSize );


                            for(size_t b = 0;b < batch_size;b ++){
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


                            size_t ndim = input[0].shape().size();
                            size_t c = input[0].shape()[ndim - 3];
                            size_t h = input[0].shape()[ndim - 2];
                            size_t w = input[0].shape()[ndim - 1];
                            size_t batchstep = h * w * c;
                            size_t batch_size = input[0].size()/ batchstep;

                            dim3 grid_size(divroundup(w , kCudaTransposeTileSize) , divroundup(h , kCudaTransposeTileSize) , divroundup(c , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize , kCudaTransposeTileSize );


                            for(size_t b = 0;b < batch_size;b ++){
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
                            std::vector<size_t> revperm = _get_reverse_perm(perm);
                            size_t totalthreads = 1;
                            for(int i = 0;i < input[0].shape().size();i++){
                                totalthreads *= divroundup(input[0].shape()[i] , kCudaTransposeTileSize);
                            }
                            size_t tilesize = (1 << (2 * input[0].shape().size()));
                            cuda_shared_pointer<size_t> shape(input[0].shape() , Cuda);
                            cuda_shared_pointer<size_t> outstrides(result.get_strides() , Cuda);
                            cuda_shared_pointer<size_t> instrides(input[0].get_strides() , Cuda);
                            cuda_shared_pointer<size_t> cuperm(perm , Cuda);
                            cuda_shared_pointer<size_t> curevperm(revperm , Cuda);

                            _transpose_forward_kernel<<<totalthreads , tilesize , sizeof(T) * tilesize>>>(result.get() , input[0].get() , 
                                result.size() , shape.size() , shape.get() , instrides.get() , outstrides.get() , cuperm.get() , curevperm.get());


                            }
                        return result;
                    }
                    else{
                        auto instrides = input[0].get_strides();
                        instrides.push_back(input[0].size());
                        auto strides = result.get_strides();
                        size_t ndim = input[0].shape().size();
                        for(int index = 0;index<input[0].size();index+= instrides[0]){
                            std::vector<size_t> idx(ndim);
                            for(int i = 0;i<ndim;i++){
                                idx[i] = index % instrides[i+1] / instrides[i];
                            }
                            std::vector<size_t> outidx = _get_transpose_vec_rev(idx , perm);
                            size_t outindex = 0;
                            for(int i = 0;i<ndim;i++){
                                outindex += outidx[i] * strides[i];
                            }
                            result.get()[outindex] = input[0].get()[index];
                        }
                    }

                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out){
                    //return {grad_out.transpose(_get_reverse_perm(perm))};
                    std::vector<size_t> revperm = _get_reverse_perm(perm);
                    std::vector<size_t> newshape = a.shape();
                    Tensor<T> result(newshape , a.device());
                    if(result.device() == Cuda){
                        TransposeType ttype;
                        size_t eq_count = 0;
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


                            size_t ndim = grad_out.shape().size();
                            size_t m = grad_out.shape()[ndim - 2];
                            size_t n = grad_out.shape()[ndim - 1];
                            size_t batchstep = m * n;
                            size_t batch_size = grad_out.size()/ batchstep;

                            dim3 grid_size(divroundup(n , kCudaTransposeTileSize) , divroundup(m , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize);


                            for(size_t b = 0;b < batch_size;b ++){
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

                            size_t ndim = grad_out.shape().size();
                            size_t h = grad_out.shape()[ndim - 3];
                            size_t w = grad_out.shape()[ndim - 2];
                            size_t c = grad_out.shape()[ndim - 1];
                            size_t batchstep = h * w * c;
                            size_t batch_size = grad_out.size()/ batchstep;

                            dim3 grid_size(divroundup(c , kCudaTransposeTileSize) , divroundup(w , kCudaTransposeTileSize) , divroundup(h , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize , kCudaTransposeTileSize );


                            for(size_t b = 0;b < batch_size;b ++){
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

                            size_t ndim = grad_out.shape().size();
                            size_t c = grad_out.shape()[ndim - 3];
                            size_t h = grad_out.shape()[ndim - 2];
                            size_t w = grad_out.shape()[ndim - 1];
                            size_t batchstep = h * w * c;
                            size_t batch_size = grad_out.size()/ batchstep;

                            dim3 grid_size(divroundup(w , kCudaTransposeTileSize) , divroundup(h , kCudaTransposeTileSize) , divroundup(c , kCudaTransposeTileSize));
                            dim3 block_size(kCudaTransposeTileSize , kCudaTransposeTileSize , kCudaTransposeTileSize );


                            for(size_t b = 0;b < batch_size;b ++){
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
                             size_t totalthreads = 1;
                            for(int i = 0;i < grad_out.shape().size();i++){
                                totalthreads *= divroundup(grad_out.shape()[i] , kCudaTransposeTileSize);
                            }
                            size_t tilesize = (1 << (2 * grad_out.shape().size()));
                            cuda_shared_pointer<size_t> shape(grad_out.shape() , Cuda);
                            cuda_shared_pointer<size_t> outstrides(result.get_strides() , Cuda);
                            cuda_shared_pointer<size_t> instrides(grad_out.get_strides() , Cuda);
                            cuda_shared_pointer<size_t> cuperm(revperm , Cuda);
                            cuda_shared_pointer<size_t> curevperm(perm , Cuda);

                            _transpose_forward_kernel<<<totalthreads , tilesize , sizeof(T) * tilesize>>>(result.get() , grad_out.get() , 
                                result.size() , shape.size() , shape.get() , instrides.get() , outstrides.get() , cuperm.get() , curevperm.get());


                            }
                        return {result};
                    }
                    else{
                        auto instrides = grad_out.get_strides();
                        instrides.push_back(grad_out.size());
                        auto strides = result.get_strides();
                        size_t ndim = grad_out.shape().size();
                        for(int index = 0;index<grad_out.size();index+= instrides[0]){
                            std::vector<size_t> idx(ndim);
                            for(int i = 0;i<ndim;i++){
                                idx[i] = index % instrides[i+1] / instrides[i];
                            }
                            std::vector<size_t> outidx = _get_transpose_vec_rev(idx , revperm);
                            size_t outindex = 0;
                            for(int i = 0;i<ndim;i++){
                                outindex += outidx[i] * strides[i];
                            }
                            result.get()[outindex] = grad_out.get()[index];
                        }
                    }

                    return {result};
                }
                std::vector<Tensor<T>> get_inputs() const override{
                    return {a};
                }



        };


        template <typename T>
        __global__ void _pool_forward_kernel(T * result , const T * input , size_t * mask ,size_t ndim ,  const size_t * kernel_shape , 
            const size_t result_size ,  const size_t *  input_shape , const size_t * result_shape
             , const size_t * input_strides){
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if(index >= result_size)
                return;
            size_t outidx[kCudaMultiDimMax];
            size_t index_ = index;
            for(int i = 0;i<ndim;i++){
                outidx[i] = index_ % result_shape[ndim - i - 1];
                index_ /= result_shape[ndim - i - 1];
            }
            CudaMultiDimIndex kernel_idx(kernel_shape , ndim);
            do{
                size_t inputindex = 0;
                for(size_t i = 0;i<ndim;i++){
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
        __global__ void _pool_backward_kernel(T * grad_in , const T * grad_out ,const size_t * mask , 
            const size_t result_size ){
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if(index >= result_size)
                return;
            size_t inputindex = mask[index];
            grad_in[inputindex] += grad_out[index];
        }
        template <typename T>
        class Pool2dFunc : public Function<T>{
            private:
                std::vector<size_t> kernel_shape_;
                cuda_shared_pointer<size_t> mask;
                Tensor<T> a;
            public:
                Pool2dFunc(const std::vector<size_t> & kernel_shape ) : kernel_shape_(kernel_shape){}
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs){
                    // input : (* , H , W)
                    if(inputs.size() != 1){
                        throw std::runtime_error("PoolFunc error!");
                    }
                    auto input = inputs[0];
                    std::vector<size_t> single_input_shape(input.shape().end() - 2 , input.shape().end());
                    auto single_output_shape = single_input_shape;
                    size_t ndim =  input.ndim();
                    for(size_t i = 0;i< 2;i++){
                        single_output_shape[i] /= kernel_shape_[i];
                    }
                    auto resultshape = input.shape();
                    resultshape[resultshape.size() - 1] = single_output_shape[1];
                    resultshape[resultshape.size() - 2] = single_output_shape[0];
                    Tensor<T> result(resultshape , input.device());
                    mask = cuda_shared_pointer<size_t>(result.size() , input.device());
                    auto inputstrides = input.get_strides();
                    inputstrides.push_back(input.size());
                    size_t inputstep = inputstrides[2];
                    size_t resultstep = prod_vec(single_output_shape);
                    std::vector<size_t> single_input_strides(inputstrides.begin() , inputstrides.begin() + 2);
                    cuda_shared_pointer<size_t> kernel_shape_cuda(kernel_shape_ , input.device());
                    cuda_shared_pointer<size_t> single_output_shape_cuda(single_output_shape , input.device());
                    cuda_shared_pointer<size_t> single_input_strides_cuda(single_input_strides , input.device());
                    cuda_shared_pointer<size_t> single_input_shape_cuda(single_input_shape , input.device());
                    if(result.device() == Cuda){
                        for(size_t inputoffset = 0 , outputoffset = 0;inputoffset < input.size();inputoffset += inputstep , outputoffset += resultstep){
                            _pool_forward_kernel<T><<<CudaGetBlocks(resultstep) , kCudaThreadsNum>>>(
                                result.get() + outputoffset ,
                                input.get() + inputoffset , 
                                mask.get() + outputoffset , 
                                2,
                                kernel_shape_cuda.get(), 
                                resultstep,
                                single_input_shape_cuda.get() , 
                                single_output_shape_cuda.get(),
                                single_input_strides_cuda.get()
                            );
                        }
                    }
                    else{

                        for(size_t inputoffset = 0 , outputoffset = 0;inputoffset < input.size();inputoffset += inputstep , outputoffset += resultstep){
                            for(size_t i = 0;i<single_output_shape[0] * kernel_shape_[0];i++){
                                for(int j = 0;j<single_output_shape[1] * kernel_shape_[1];j++){
                                    size_t inputindex = (i * single_input_strides[1]) + (j * single_input_strides[0]);
                                    size_t resultindex = (i / kernel_shape_[0]) * single_output_shape[1] + (j / kernel_shape_[1]);
                                    if(result.get()[resultindex + outputoffset] < input.get()[inputindex + inputoffset]){
                                        result.get()[resultindex + outputoffset] = input.get()[inputindex + inputoffset];
                                        mask.get()[resultindex + outputoffset] = inputindex;
                                    }
                                }
                            }
                        }
                    }
                    if(input.requires_grad()){
                        result.set_requires_grad(true);
                        a = input;
                    }
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out){
                    // input : (* , H , W)
                    auto input = a;
                    std::vector<size_t> single_input_shape(input.shape().end() - 2 , input.shape().end());
                    Tensor<T> grad_in(input.shape() , input.device());
                    auto inputstrides = input.get_strides();
                    std::vector<size_t> single_output_shape(grad_out.shape().end() - 2 , grad_out.shape().end());
                    inputstrides.push_back(input.size());
                    size_t inputstep = inputstrides[2];
                    size_t resultstep = prod_vec(single_output_shape);
                    std::vector<size_t> single_input_strides(inputstrides.begin() , inputstrides.begin() + 2);

                    size_t single_output_size = prod_vec(single_output_shape);
                    if(grad_in.device() == Cuda){
                        for(size_t inputoffset = 0 , outputoffset = 0;inputoffset < input.size();inputoffset += inputstep , outputoffset += resultstep){
                            _pool_backward_kernel<T><<<CudaGetBlocks(resultstep) , kCudaThreadsNum>>>(
                                grad_in.get() + inputoffset,grad_out.get() + outputoffset,
                                mask.get() + outputoffset,
                                single_output_size
                            );
                        }
                    }
                    else{
                        for(size_t inputoffset = 0 , outputoffset = 0;inputoffset < input.size();inputoffset += inputstep , outputoffset += resultstep){
                            for(size_t i = 0;i<single_output_shape[0];i++){
                                for(size_t j = 0;j<single_output_shape[1];j++){
                                    size_t inputindex = mask.get()[outputoffset + i * single_output_shape[1] + j];
                                    grad_in.get()[inputindex + inputoffset] += grad_out.get()[outputoffset + i * single_output_shape[1] + j];
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
        class ReshapeFunc : public Function<T>{
            private:
                std::vector<size_t> newshape;
                std::vector<size_t> oldshape;
                Tensor<T> a;
            public:
                ReshapeFunc(const std::vector<size_t> & newshape) : newshape(newshape){}
                Tensor<T> forward(const std::vector<Tensor<T>> & input ) override{
                    oldshape = input[0].shape();
                    if(input.size() != 1 )
                        throw std::runtime_error("ReshapeFunc error!");
                    if(input[0].requires_grad()){
                        a = input[0];
                    }
                    Tensor<T> result(newshape , input[0].device());
                    result.get_shared_ptr() = input[0].get_shared_ptr();
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
                    std::vector<size_t> newshape = input[0].shape();
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
                    size_t step0 = input0stride[2];
                    size_t step1 = input1stride[2];
                    size_t stepresult = resultstride[2];
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
                    std::vector<size_t> gradperm;
                    size_t ndim =  a.shape().size();
                    for(int i = 0;i<ndim ;i++){
                        gradperm.push_back(i);
                    }
                    std::swap(gradperm[ndim - 2] , gradperm[ndim - 1]);
                    result.push_back(grad_out.matmul(b.transpose(gradperm)));
                    result.push_back(a.transpose(gradperm).matmul(grad_out));
                    return result;
                }
                std::vector<Tensor<float>> get_inputs() const override{
                    return {a , b};
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
                    std::vector<size_t> newshape = input[0].shape();
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
                    size_t step0 = input0stride[2];
                    size_t step1 = input1stride[2];
                    size_t stepresult = resultstride[2];
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
                    std::vector<size_t> gradperm;
                    size_t ndim =  a.shape().size();
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
        template<typename T>
        __global__ void im2col_gpu_nopadding_t(T * col , const T * im , 
            const size_t kernel_size , const size_t ndim , 
            const size_t * kernel_shape ,  const size_t * imshape , 
            const size_t imsize , const size_t reduce_imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < reduce_imsize){
                size_t grid_min[kCudaMultiDimMax]; // index in image
                for(size_t i = 0 , index_ = index;i<ndim;i++){
                    size_t reduce_imshape = imshape[ndim - i - 1] - (kernel_shape[ndim - i - 1] >> 1) - 1;
                    grid_min[ndim - 1 - i] = index_ % reduce_imshape;
                    index_ /= reduce_imshape;
                }

                size_t grid_offset =  0;
                size_t col_offset = index;
                size_t kernel_index[kCudaMultiDimMax];
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                    }
                    size_t im_offset = 0;
                    for(int i = 0;i<ndim;i++){
                        im_offset *= imshape[i];
                        im_offset += kernel_index[i];
                    }
                    col[col_offset] =  im[im_offset];
                    grid_index.next();
                    grid_offset++;
                    col_offset+=reduce_imsize;

                }while(!grid_index.is_zero());

            }
        }

        template<typename T>
        __global__ void im2col_gpu_nopadding_2d_t(T * col , const T * im , 
            const size_t kernel_size ,
            const size_t * kernel_shape ,  const size_t * imshape , 
            const size_t imsize , const size_t reduce_imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < reduce_imsize){
                size_t grid_min[2]; // index in image
                size_t reduce_imshape = imshape[1] - (kernel_shape[1] >> 1) - 1;
                grid_min[1] = index % reduce_imshape;
                grid_min[0] = index / reduce_imshape;


                size_t grid_offset =  0;
                size_t col_offset = index;
                size_t kernel_index[2];
                size_t grid_index[2];
                for(grid_index[0] = 0;grid_index[0] < kernel_shape[0] ; grid_index[0] ++ ){
                    for(grid_index[1] = 0;grid_index[1] < kernel_shape[1]; grid_index[1] ++){
                        kernel_index[0] = grid_min[0] + grid_index[0];
                        kernel_index[1] = grid_min[1] + grid_index[1];
                        size_t im_offset = kernel_index[0] * imshape[1] + kernel_index[1];
                        col[col_offset] =  im[im_offset];
                        grid_offset++;
                        col_offset+=reduce_imsize;
                    } 
                }

            }
        }

        template<typename T>
        __global__ void im2col_gpu_nopadding(T * col , const T * im , 
            const size_t kernel_size , const size_t ndim , 
            const size_t * kernel_shape ,  const size_t * imshape , 
            const size_t imsize , const size_t reduce_imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < reduce_imsize){
                size_t grid_min[kCudaMultiDimMax]; // index in image
                for(size_t i = 0 , index_ = index;i<ndim;i++){
                    size_t reduce_imshape = imshape[ndim - i - 1] - (kernel_shape[ndim - i - 1] >> 1) - 1;
                    grid_min[ndim - 1 - i] = index_ % reduce_imshape;
                    index_ /= reduce_imshape;
                }

                size_t grid_offset =  0;
                size_t col_offset = index * kernel_size;
                size_t kernel_index[kCudaMultiDimMax];
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                    }
                    size_t im_offset = 0;
                    for(int i = 0;i<ndim;i++){
                        im_offset *= imshape[i];
                        im_offset += kernel_index[i];
                    }
                    col[col_offset] =  im[im_offset];
                    grid_index.next();
                    grid_offset++;
                    col_offset++;

                }while(!grid_index.is_zero());

            }
        }

        template<typename T>
        __global__ void im2col_gpu_nopadding_2d(T * col , const T * im , 
            const size_t kernel_size ,
            const size_t * kernel_shape ,  const size_t * imshape , 
            const size_t imsize , const size_t reduce_imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < reduce_imsize){
                size_t grid_min[2]; // index in image
                size_t reduce_imshape = imshape[1] - (kernel_shape[1] >> 1) - 1;
                grid_min[1] = index % reduce_imshape;
                grid_min[0] = index / reduce_imshape;


                size_t grid_offset =  0;
                size_t col_offset = index * kernel_size;
                size_t kernel_index[2];
                size_t grid_index[2];
                for(grid_index[0] = 0;grid_index[0] < kernel_shape[0] ; grid_index[0] ++ ){
                    for(grid_index[1] = 0;grid_index[1] < kernel_shape[1]; grid_index[1] ++){
                        kernel_index[0] = grid_min[0] + grid_index[0];
                        kernel_index[1] = grid_min[1] + grid_index[1];
                        size_t im_offset = kernel_index[0] * imshape[1] + kernel_index[1];
                        col[col_offset] =  im[im_offset];
                        grid_offset++;
                        col_offset++;
                    } 
                }

            }
        }
        template<typename T>
        __global__ void im2col_gpu(T * col , const T * im , const size_t kernel_size , const size_t ndim , const size_t * kernel_shape ,  const size_t * imshape , const size_t imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < imsize){
                size_t imidx[kCudaMultiDimMax];
                for(int i = 0 , index_ = index;i<ndim;i++){
                    imidx[i] = index_ % imshape[ndim - i - 1];
                    index_ /= imshape[ndim - i - 1];
                }
                size_t grid_min[kCudaMultiDimMax];
                for(int i = 0;i<ndim;i++){
                    grid_min[i] = imidx[ndim - i - 1] - kernel_shape[i] / 2;
                }
                size_t grid_offset =  0;
                size_t col_offset = index * kernel_size;
                size_t kernel_index[kCudaMultiDimMax];
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    bool is_valid = true;
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                        if( kernel_index[i] >= imshape[i]){
                            is_valid = false;
                            break;
                        }
                    }
                    size_t im_offset = 0;
                    for(int i = 0;i<ndim;i++){
                        im_offset *= imshape[i];
                        im_offset += kernel_index[i];
                    }
                    col[col_offset] =  is_valid ? im[im_offset] : 0;
                    grid_index.next();
                    grid_offset++;
                    col_offset++;

                }while(!grid_index.is_zero());

            }
        }

        template<typename T>
        __global__ void im2col_gpu_2d(T * col , const T * im  , const size_t kernel_size 
            , const size_t * kernel_shape ,  const size_t * imshape 
            , const size_t imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < imsize){
                size_t imidx[2];
                imidx[0] = index % imshape[1];
                imidx[1] = index / imshape[1];
                size_t grid_min[2];
                grid_min[0] = imidx[1] - (kernel_shape[0] >> 1);
                grid_min[1] = imidx[0] - (kernel_shape[1] >> 1);
                size_t grid_offset =  0;
                size_t col_offset = index * kernel_size;
                size_t grid_index0 , grid_index1;
                for(grid_index0 = 0;grid_index0 < kernel_shape[0];grid_index0++){
                    for(grid_index1 = 0;grid_index1 < kernel_shape[1];grid_index1++)
                    {
                        bool is_valid = true;
                        size_t kernel_index[2];
                        kernel_index[0] = grid_min[0] + grid_index0;
                        kernel_index[1] = grid_min[1] + grid_index1;
                        is_valid = kernel_index[0] < imshape[0] && kernel_index[1] < imshape[1];
                        size_t im_offset = kernel_index[0] * imshape[1] + kernel_index[1];
                        col[col_offset] =  is_valid ? im[im_offset] : 0;
                        grid_offset++;
                        col_offset++;
                    }
                }

            }
        }
        template<typename T>
        __global__ void im2col_gpu_t(T * col , const T * im , const size_t kernel_size , const size_t ndim , const size_t * kernel_shape ,  const size_t * imshape , const size_t imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < imsize){
                size_t imidx[kCudaMultiDimMax];
                for(int i = 0 , index_ = index;i<ndim;i++){
                    imidx[i] = index_ % imshape[ndim - i - 1];
                    index_ /= imshape[ndim - i - 1];
                }
                size_t grid_min[kCudaMultiDimMax];
                for(int i = 0;i<ndim;i++){
                    grid_min[i] = imidx[ndim - i - 1] - kernel_shape[i] / 2;
                }
                size_t grid_offset =  0;
                size_t col_offset = index;
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    bool is_valid = true;
                    size_t kernel_index[kCudaMultiDimMax];
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                        if( kernel_index[i] >= imshape[i]){
                            is_valid = false;
                            break;
                        }
                    }
                    size_t im_offset = 0;
                    for(int i = 0;i<ndim;i++){
                        im_offset *= imshape[i];
                        im_offset += kernel_index[i];
                    }
                    col[col_offset] =  is_valid ? im[im_offset] : 0;
                    grid_index.next();
                    grid_offset++;
                    col_offset+=imsize;

                }while(!grid_index.is_zero());

            }
        }

        template<typename T>
        __global__ void im2col_gpu_2d_t(T * col , const T * im  , const size_t kernel_size 
            , const size_t * kernel_shape ,  const size_t * imshape 
            , const size_t imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < imsize){
                size_t imidx[2];
                imidx[0] = index % imshape[1];
                imidx[1] = index / imshape[1];
                size_t grid_min[2];
                grid_min[0] = imidx[1] - (kernel_shape[0] >> 1);
                grid_min[1] = imidx[0] - (kernel_shape[1] >> 1);
                size_t grid_offset =  0;
                size_t col_offset = index;
                size_t grid_index[2];
                for(grid_index[0] = 0;grid_index[0] < kernel_shape[0];grid_index[0]++){
                    for(grid_index[1] = 0;grid_index[1] < kernel_shape[1];grid_index[1]++)
                    {
                        bool is_valid = true;
                        size_t kernel_index[2];
                        kernel_index[0] = grid_min[0] + grid_index[0]; 
                        kernel_index[1] = grid_min[1] + grid_index[1];
                        is_valid = kernel_index[0] < imshape[0] && kernel_index[1] < imshape[1];
                        size_t im_offset = kernel_index[0] * imshape[1] + kernel_index[1];
                        col[col_offset] =  is_valid ? im[im_offset] : 0;
                        grid_offset++;
                        col_offset+= imsize;
                    }
                }

            }
        }


        template <typename T>
        void im2col_ptr(T * output , const T * input , const std::vector<size_t> & inputshape  , const std::vector<size_t> & kernel_shape , 
            const std::vector<size_t> & inputstride , const std::vector<size_t> & outputstride , const Device device){
            if(kernel_shape.size()!= inputshape.size())
                throw std::runtime_error("im2col: kernel_shape size must be input shape size ");
            auto half_kernel_shape = kernel_shape;
            auto instride = inputstride;
            auto revinstride = instride;
            auto kernel_size = prod_vec(kernel_shape);
            size_t inputsize = prod_vec(inputshape);
            std::reverse(revinstride.begin() , revinstride.end());
            for(int i = 0;i<half_kernel_shape.size();i++){
                half_kernel_shape[i] /= 2;
            }
            std::vector<size_t> kernel_stride = {};
            size_t kernel_stride_ = 1;
            for(int i = 0;i<kernel_shape.size();i++){
                kernel_stride.push_back(kernel_stride_);
                kernel_stride_ *= kernel_shape[kernel_shape.size() - 1 - i];
            }
            if(device == Device::Cpu){
                MultiDimIndex index(inputshape);
                do{
                    auto grid_min = index.get_index();
                    for(int i = 0;i< grid_min.size();i++){
                        grid_min[i] -= half_kernel_shape[i];
                    }
                    MultiDimIndex grid_index(kernel_shape);
                    do{
                        bool is_valid = true;
                        std::vector<size_t> kernel_index(grid_index.get_index());
                        for(int i = 0;i<kernel_index.size();i++){
                            kernel_index[i] += grid_min[i];
                            if( kernel_index[i] >= inputshape[i]){
                                is_valid = false;
                                break;
                            }
                        }
                        if(is_valid){
                            size_t input_index = dot_vec(kernel_index , revinstride);
                            size_t result_index = index.calculate_offset(instride) * kernel_size + grid_index.calculate_offset(kernel_stride);
                            output[result_index] = input[input_index];
                        }
                        grid_index.next();
                    }while(!grid_index.is_zero());
                    index.next();
                }while(!index.is_zero());
            }
            else{
                cuda_shared_pointer<size_t> kershape(kernel_shape , Cuda);
                cuda_shared_pointer<size_t> imshape(inputshape ,Cuda);
                im2col_gpu<<<CudaGetBlocks(inputsize) , kCudaThreadsNum>>>(output , input , prod_vec(kernel_shape) , inputshape.size()
                ,kershape.get() , imshape.get() ,prod_vec(inputshape));
            }
        }
        template <typename T>
        void col2im_ptr( T * input ,const T * output, const std::vector<size_t> & inputshape  , const std::vector<size_t> & kernel_shape , 
            const std::vector<size_t> & inputstride , const std::vector<size_t> & outputstride , const Device device){
            if(kernel_shape.size()!= inputshape.size())
                throw std::runtime_error("im2col: kernel_shape size must be input shape size ");
            auto half_kernel_shape = kernel_shape;
            auto instride = inputstride;
            auto revinstride = instride;
            auto kernel_size = prod_vec(kernel_shape);
            size_t inputsize = prod_vec(inputshape);
            std::reverse(revinstride.begin() , revinstride.end());
            for(int i = 0;i<half_kernel_shape.size();i++){
                half_kernel_shape[i] /= 2;
            }
            std::vector<size_t> kernel_stride = {};
            size_t kernel_stride_ = 1;
            for(int i = 0;i<kernel_shape.size();i++){
                kernel_stride.push_back(kernel_stride_);
                kernel_stride_ *= kernel_shape[kernel_shape.size() - 1 - i];
            }
            if(device == Device::Cpu){
                MultiDimIndex index(inputshape);
                do{
                    auto grid_min = index.get_index();
                    for(int i = 0;i< grid_min.size();i++){
                        grid_min[i] -= half_kernel_shape[i];
                    }
                    MultiDimIndex grid_index(kernel_shape);
                    do{
                        bool is_valid = true;
                        std::vector<size_t> kernel_index(grid_index.get_index());
                        for(int i = 0;i<kernel_index.size();i++){
                            kernel_index[i] += grid_min[i];
                            if( kernel_index[i] >= inputshape[i]){
                                is_valid = false;
                                break;
                            }
                        }
                        if(is_valid){
                            size_t input_index = dot_vec(kernel_index , revinstride);
                            size_t result_index = index.calculate_offset(instride) * kernel_size + grid_index.calculate_offset(kernel_stride);
                            input[input_index] += output[result_index];
                        }
                        grid_index.next();
                    }while(!grid_index.is_zero());
                    index.next();
                }while(!index.is_zero());
            }
            else{
                cuda_shared_pointer<size_t> kershape(kernel_shape , Cuda);
                cuda_shared_pointer<size_t> imshape(inputshape ,Cuda);
                col2im_gpu<<<CudaGetBlocks(inputsize) , kCudaThreadsNum>>>(input , output , prod_vec(kernel_shape) , inputshape.size()
                ,kershape.get() , imshape.get() ,prod_vec(inputshape));
            }
        }


        template<typename T>
        __global__ void col2im_gpu(T * im , const T * col  , const size_t kernel_size 
            , const size_t ndim , const size_t * kernel_shape ,  const size_t * imshape 
            , const size_t imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < imsize){
                size_t imidx[kCudaMultiDimMax];
                for(int i = 0 , index_ = index;i<ndim;i++){
                    imidx[i] = index_ % imshape[ndim - i - 1];
                    index_ /= imshape[ndim - i - 1];
                }
                size_t grid_min[kCudaMultiDimMax];
                for(int i = 0;i<ndim;i++){
                    grid_min[i] = imidx[ndim - i - 1] - kernel_shape[i] / 2;
                }
                size_t grid_offset =  0;
                size_t col_offset = index * kernel_size;
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    bool is_valid = true;
                    size_t kernel_index[kCudaMultiDimMax];
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                        if( kernel_index[i] >= imshape[i]){
                            is_valid = false;
                            break;
                        }
                    }
                    if(is_valid){
                        size_t im_offset = 0;
                        for(int i = 0;i<ndim;i++){
                            im_offset *= imshape[i];
                            im_offset += kernel_index[i];
                        }
                        atomicAdd( &im[im_offset] , col[col_offset] );
                    }
                    grid_index.next();
                    grid_offset++;
                    col_offset++;

                }while(!grid_index.is_zero());

            }
        }

        template<typename T>
        __global__ void col2im_gpu_2d(T * im , const T * col  , const size_t kernel_size 
            , const size_t * kernel_shape ,  const size_t * imshape 
            , const size_t imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < imsize){
                size_t imidx[2];
                imidx[0] = index % imshape[1];
                imidx[1] = index / imshape[1];
                size_t grid_min[2];
                grid_min[0] = imidx[1] - (kernel_shape[0] >> 1);
                grid_min[1] = imidx[0] - (kernel_shape[1] >> 1);
                size_t grid_offset =  0;
                size_t col_offset = index * kernel_size;
                size_t grid_index[2];
                for(grid_index[0] = 0;grid_index[0] < kernel_shape[0];grid_index[0]++){
                    for(grid_index[1] = 0;grid_index[1] < kernel_shape[1];grid_index[1]++)
                    {
                        size_t kernel_index[2];
                        bool is_valid = true;
                        kernel_index[0] = grid_min[0] + grid_index[0]; 
                        kernel_index[1] = grid_min[1] + grid_index[1];
                        if( kernel_index[0] >= imshape[0] || kernel_index[1] >= imshape[1]){
                            is_valid = false;
                        }
                        if(is_valid){
                            size_t im_offset = kernel_index[0] * imshape[1] + kernel_index[1];
                            atomicAdd( &im[im_offset] , col[col_offset] );
                        }
                        grid_offset++;
                        col_offset++;
                    }
                }

            }
        }


        template<typename T>
        __global__ void col2im_gpu_nopadding(T * im , const T * col , 
            const size_t kernel_size , const size_t ndim , 
            const size_t * kernel_shape ,  const size_t * imshape , 
            const size_t imsize , const size_t reduce_imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < reduce_imsize){
                size_t grid_min[kCudaMultiDimMax]; // index in image
                for(size_t i = 0 , index_ = index;i<ndim;i++){
                    size_t reduce_imshape = imshape[ndim - i - 1] - (kernel_shape[ndim - i - 1] >> 1) - 1;
                    grid_min[ndim - 1 - i] = index_ % reduce_imshape;
                    index_ /= reduce_imshape;
                }

                size_t grid_offset =  0;
                size_t col_offset = index * kernel_size;
                size_t kernel_index[kCudaMultiDimMax];
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                    }
                    size_t im_offset = 0;
                    for(int i = 0;i<ndim;i++){
                        im_offset *= imshape[i];
                        im_offset += kernel_index[i];
                    }
                    atomicAdd( &im[im_offset] , col[col_offset] );
                    grid_index.next();
                    grid_offset++;
                    col_offset++;

                }while(!grid_index.is_zero());

            }
        }

        template<typename T>
        __global__ void col2im_gpu_nopadding_2d(T * im , const T * col , 
            const size_t kernel_size ,
            const size_t * kernel_shape ,  const size_t * imshape , 
            const size_t imsize , const size_t reduce_imsize){
            size_t index = threadIdx.x + blockDim.x * blockIdx.x;
            if(index < reduce_imsize){
                size_t grid_min[2]; // index in image
                size_t reduce_imshape = imshape[1] - (kernel_shape[1] >> 1) - 1;
                grid_min[1] = index % reduce_imshape;
                grid_min[0] = index / reduce_imshape;


                size_t grid_offset =  0;
                size_t col_offset = index * kernel_size;
                size_t kernel_index[2];
                size_t grid_index[2];
                for(grid_index[0] = 0;grid_index[0] < kernel_shape[0] ; grid_index[0] ++ ){
                    for(grid_index[1] = 0;grid_index[1] < kernel_shape[1]; grid_index[1] ++){
                        kernel_index[0] = grid_min[0] + grid_index[0];
                        kernel_index[1] = grid_min[1] + grid_index[1];
                        size_t im_offset = kernel_index[0] * imshape[1] + kernel_index[1];
                        atomicAdd( &im[im_offset] , col[col_offset] );
                        grid_offset++;
                        col_offset++;
                    } 
                }

            }
        }




    }
    template <typename T>
    void __cudaMemcpyBatch(T * dst , const T * src ,const size_t size , const size_t batch_size , cudaStream_t * streams ){

        for(size_t b = 0;b < batch_size;b ++){
            int s = b % kStreamCount;
            cudaMemcpyAsync(dst + b * size, src , size * sizeof(T) , cudaMemcpyDeviceToDevice , streams[s]);

        }
    }
    template <typename T>
    __global__ void _linear_add(T * output , const T * input , size_t size){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < size){
            output[index] += input[index];
        }
    }
    template <typename T>
    class Linear : public Module<T>{
        private:
            Tensor<T> weight;
            Tensor<T> weight_t;
            Tensor<T> bias;
            Tensor<T> input_cache; // internal backward
            cudaStream_t stream_gemm,stream_add;
            cublasHandle_t handle;
            cudaStream_t streams[kStreamCount];
        public:
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
                cublasCreate(&handle);
                cudaStreamCreate(&stream_gemm);
                cudaStreamCreate(&stream_add);
                cublasSetStream(handle , stream_gemm);
                for(int i = 0;i<kStreamCount;i++){
                    CHECK(cudaStreamCreate(&streams[i]));
                }
            }
            Linear(const size_t in_features, const size_t out_features, Device device = DefaultDevice){
                weight = randn<T>({out_features , in_features} , device) * rsqrt(in_features / (T)2);
                bias = randn<T>({out_features} , device) ;
                cublasCreate(&handle);
                cudaStreamCreate(&stream_gemm);
                cudaStreamCreate(&stream_add);
                cublasSetStream(handle , stream_gemm);
                for(int i = 0;i<kStreamCount;i++){
                    CHECK(cudaStreamCreate(&streams[i]));
                }
            }
            ~Linear(){
                cublasDestroy(handle);
                cudaStreamDestroy(stream_gemm);
                cudaStreamDestroy(stream_add);
                for(int i = 0;i<kStreamCount;i++){
                    CHECK(cudaStreamDestroy(streams[i]));
                }
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                if(inputs.size() != 1){
                    throw std::runtime_error("Linear: input size must be 1");
                }
                auto input = inputs[0];
                if(input.shape().back() != weight.shape().back()){
                    throw std::runtime_error("Linear: input shape and weight shape mismatch");
                }
                if(input.requires_grad()){
                    input_cache = input;
                }
                std::vector<size_t> newshape = input.shape();
                newshape.pop_back();
                newshape.push_back(bias.size());
                Tensor<T> result(newshape , input.device());
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
                size_t step1 = input1stride[1];
                size_t stepresult = resultstride[1];
                if(result.device() == Cuda){
                    weight.to(Cuda);
                    bias.to(Cuda);
                    T alpha = 1.0f;
                    T beta = 1.0f;
                    __cudaMemcpyBatch(result.get() , bias.get() , stepresult , result.size() / stepresult , streams);

                    auto batch_size = result.size() / stepresult;

                    if constexpr (std::is_same_v<T , float>){
                        CHECK_CUBLAS(cublasSgemmStridedBatched(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                            1,
                            resultshape[resultshape.size() - 1],
                            input0shape[input0shape.size() - 1],
                            &alpha,
                            input.get(),
                            1,
                            step1,
                            weight.get(),
                            input0stride[1],
                            0,
                            &beta,
                            result.get(),
                            1,
                            stepresult,
                            batch_size
                        ));
                    }
                    else if constexpr (std::is_same_v<T , double>){
                        CHECK_CUBLAS(cublasDgemmStridedBatched(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                            1,
                            resultshape[resultshape.size() - 1],
                            input0shape[input0shape.size() - 1],
                            &alpha,
                            input.get(),
                            1,
                            step1,
                            weight.get(),
                            input0stride[1],
                            0,
                            &beta,
                            result.get(),
                            1,
                            stepresult,
                            batch_size
                        ));
                    }


                }
                else{
                    weight.to(Cpu);
                    bias.to(Cpu);
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
                return result;
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                weight_t = weight.transpose({1 , 0});
                auto input = input_cache;
                Tensor<T> grad_input(input.shape() , input.device());
                Tensor<T> grad_weight(weight.shape() , weight.device());
                Tensor<T> grad_bias(bias.shape() , bias.device());
                auto inputstrides = input.get_strides();
                inputstrides.push_back(input.size());
                auto gradoutstrides = grad_out.get_strides();
                gradoutstrides.push_back(grad_out.size());
                size_t stepinput = inputstrides[1];
                size_t stepgradout = gradoutstrides[1];
                if(input.device() == Cuda){
                    T alpha = 1.0f;
                    T beta = 1.0f;



                    for(size_t inputoffset = 0 , gradoutoffset = 0;gradoutoffset < grad_out.size();inputoffset += stepinput , gradoutoffset += stepgradout){
                        
                        if constexpr (std::is_same_v<T , float>){
                            CHECK_CUBLAS(
                            cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                input.shape().back() ,  grad_out.shape().back(), 1,
                                &alpha , 
                                input.get() + inputoffset , 
                                stepinput,
                                grad_out.get() + gradoutoffset,
                                1
                                ,&beta , 
                                grad_weight.get(),
                                grad_weight.shape().back()
                            ));
                        }
                        else if constexpr (std::is_same_v<T , double>){
                            CHECK_CUBLAS(
                            cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                input.shape().back() ,  grad_out.shape().back(), 1,
                                &alpha , 
                                input.get() + inputoffset , 
                                stepinput,
                                grad_out.get() + gradoutoffset,
                                1
                                ,&beta , 
                                grad_weight.get(),
                                grad_weight.shape().back()
                            ));
                        }

                        


                        _linear_add<T><<<CudaGetBlocks(grad_bias.size()) , kCudaThreadsNum , 0 , stream_add>>>(
                            grad_bias.get() , 
                            grad_out.get() + gradoutoffset , 
                            grad_bias.size()
                        );

                    }

                    
                    auto batch_size = grad_out.size() / stepgradout;
                    if constexpr (std::is_same_v<T , float>){
                        CHECK_CUBLAS(
                        cublasSgemmStridedBatched(handle , CUBLAS_OP_N, CUBLAS_OP_N,
                            1, input.shape().back(), grad_out.shape().back(),
                            &alpha,
                            grad_out.get(),
                            1,
                            stepgradout,
                            weight_t.get(),
                            weight_t.shape().back(),
                            0,
                            &beta,
                            grad_input.get(),
                            1,
                            stepinput,
                            grad_out.size() / stepgradout
                        ));
                    }
                    else if constexpr (std::is_same_v<T , double>){
                        CHECK_CUBLAS(
                        cublasDgemmStridedBatched(handle , CUBLAS_OP_N, CUBLAS_OP_N,
                            1, input.shape().back(), grad_out.shape().back(),
                            &alpha,
                            grad_out.get(),
                            1,
                            stepgradout,
                            weight_t.get(),
                            weight_t.shape().back(),
                            0,
                            &beta,
                            grad_input.get(),
                            1,
                            stepinput,
                            grad_out.size() / stepgradout
                        ));
                    }

                }
                else{
                    for(size_t inputoffset = 0 , gradoutoffset = 0;gradoutoffset < grad_out.size();inputoffset += stepinput , gradoutoffset += stepgradout){
                        for(size_t i = 0;i<grad_bias.size();i++){
                            grad_bias.get()[i] += grad_out.get()[gradoutoffset + i];
                        }
                        for(size_t i = 0;i< grad_input.shape().back();i++){
                            for(size_t k = 0;k< grad_out.shape().back();k++){
                                grad_input.get()[inputoffset + i] += weight.get()[i + k * grad_input.shape().back()] * 
                                    grad_out.get()[gradoutoffset + k];
                            }
                        }
                        for(size_t i = 0;i< grad_weight.shape()[grad_weight.ndim() - 2];i++){
                            for(size_t j = 0;j< grad_weight.shape()[grad_weight.ndim() - 1];j++){
                                grad_weight.get()[i * grad_weight.shape().back() + j] += 
                                    grad_out.get()[gradoutoffset + i] * 
                                    input.get()[inputoffset + j];
                            }
                        }
                    }
                }
                weight.set_grad(grad_weight);
                bias.set_grad(grad_bias);

                return {grad_input};
            }
            std::vector<Tensor<T>> parameters() override{
                return {weight , bias};
            }
    };

    enum PaddingMode{
        NoPadding,
        ZeroPadding
    };

    template <typename T>
    class Conv : public Module<T>{
        private:
            Tensor<T> kernel_;
            Tensor<T> input_cache;
            PaddingMode padding_mode;
            std::vector<size_t> single_kernel_shape;
            std::vector<size_t> kernelstride;
            size_t single_kernel_size;
            std::vector<size_t> half_kernel_shape;
            T * im2col_buf[2];
            size_t im2col_size_prev;
            cublasHandle_t handle;
            cudaStream_t stream_im2col, stream_gemm;
            cudaEvent_t ev[2];
        public:
            ~Conv(){
                if(im2col_buf[0] != nullptr){
                    CHECK(cudaFree(im2col_buf[0]));
                    CHECK(cudaFree(im2col_buf[1]));
                }
                CHECK(cudaStreamSynchronize(stream_gemm));
                CHECK(cudaStreamSynchronize(stream_im2col));
                CHECK(cudaStreamDestroy(stream_gemm));
                CHECK(cudaStreamDestroy(stream_im2col));
                cublasDestroy(handle);
            }

            Conv(const Tensor<T> & kernel , PaddingMode padding_mode = ZeroPadding)
            : kernel_(kernel) , padding_mode(padding_mode) {
                im2col_buf[0] = nullptr;
                im2col_buf[1] = nullptr;
                im2col_size_prev = 0;
                cublasCreate(&handle);
                cudaStreamCreate(&stream_im2col);
                cudaStreamCreate(&stream_gemm);
                cudaEventCreate(&ev[0]);
                cudaEventCreate(&ev[1]);
                cublasSetStream(handle , stream_gemm);
                kernelstride = kernel_.get_strides();
                single_kernel_shape = std::vector<size_t>(kernel_.shape().begin()+2 , kernel_.shape().end());
                single_kernel_size = Functional::prod_vec(single_kernel_shape);
                half_kernel_shape = single_kernel_shape;
                for(int i = 0;i<half_kernel_shape.size();i++){
                    half_kernel_shape[i] /= 2;
                }
            }
            Conv(const std::vector<size_t> & kernel_shape,
                PaddingMode padding_mode = ZeroPadding , Device device = DefaultDevice) : padding_mode(padding_mode) {
                im2col_buf[0] = nullptr;
                im2col_buf[1] = nullptr;
                im2col_size_prev = 0;
                cublasCreate(&handle);
                cudaStreamCreate(&stream_im2col);
                cudaStreamCreate(&stream_gemm);
                cudaEventCreate(&ev[0]);
                cudaEventCreate(&ev[1]);
                cublasSetStream(handle , stream_gemm);
                single_kernel_shape = std::vector<size_t>(kernel_shape.begin()+2 , kernel_shape.end());
                single_kernel_size = Functional::prod_vec(single_kernel_shape);
                kernel_ = randn<T>(kernel_shape , device) * rsqrt(single_kernel_size / (T)2);
                kernelstride = kernel_.get_strides();
                half_kernel_shape = single_kernel_shape;
                for(int i = 0;i<half_kernel_shape.size();i++){
                    half_kernel_shape[i] /= 2;
                }
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                auto input = inputs[0];
                // input :(N , C_in , ...) kernel (C_out , C_in , ...) result (N , C_out , ...)
                if(input.ndim() != kernel_.ndim()){
                    std::cerr << "Conv: input and kernel must have the same number of dimensions" << std::endl;
                    throw std::runtime_error("Conv: input and kernel must have the same number of dimensions");
                }
                if(input.shape()[1] != kernel_.shape()[1]){
                    std::cerr << "Conv: input and kernel shape mismatch" << std::endl;
                    throw std::runtime_error("Conv: input and kernel must have the same number of input channels");
                }
                if(input.requires_grad()){
                    input_cache = input;
                }
                kernel_.to(input.device());
                auto resultshape = input.shape();
                resultshape[1] = kernel_.shape()[0];

                if(padding_mode == NoPadding){
                    for(size_t i = 2;i<resultshape.size();i++){
                        resultshape[i] -= (kernel_.shape()[i] - (kernel_.shape()[i] & 1));
                    }
                }

                Tensor<T> result(resultshape , input.device());

                std::vector<size_t> resultstride = result.get_strides();
                auto inputstride = input.get_strides();
                std::vector<size_t> single_input_shape(input.shape().begin()+2 , input.shape().end());
                std::vector<size_t> single_input_stride(inputstride.begin() , inputstride.end() - 2);
                if(result.device() == Cuda){
                    //ready for pipeline


                    size_t single_input_size = Functional::prod_vec(single_input_shape);
                    size_t im2col_size = single_input_size * single_kernel_size * sizeof(T);
                    if(im2col_buf[0] == nullptr){
                        CHECK(cudaMalloc(&im2col_buf[0] , im2col_size));
                        CHECK(cudaMalloc(&im2col_buf[1] , im2col_size));
                        im2col_size_prev = im2col_size;
                    }
                    else if(im2col_size_prev != im2col_size){
                        CHECK(cudaFree(im2col_buf[0]));
                        CHECK(cudaFree(im2col_buf[1]));
                        CHECK(cudaMalloc(&im2col_buf[0] , im2col_size));
                        CHECK(cudaMalloc(&im2col_buf[1] , im2col_size));
                        im2col_size_prev = im2col_size;
                    }

                    T alpha = 1.0f;
                    T beta = 1.0f;

                    size_t iter = 0;

                    //preprocessing
                    auto instride = single_input_stride;
                    auto revinstride = instride;
                    std::reverse(revinstride.begin() , revinstride.end());
                    cuda_shared_pointer<size_t> kershape(single_kernel_shape , Cuda);
                    cuda_shared_pointer<size_t> imshape(single_input_shape ,Cuda);
                    size_t inputbatchoffset , resultbatchoffset , resultoutoffset , kerneloutoffset , inputinoffset , kernelinoffset;
                    size_t prev_kerneloutoffset , prev_kernelinoffset , prev_resultoutoffset , prev_resultbatchoffset;
                    if(padding_mode == ZeroPadding){

                        if(single_kernel_shape.size() == 2){
                            for(inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                                for(resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                    ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                    for(inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        Functional::im2col_gpu_2d<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>
                                        (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                        single_kernel_size
                                        ,kershape.get() , imshape.get() ,single_input_size);
                                        CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                        
                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                            if constexpr (std::is_same_v<T , float>){
                                                CHECK_CUBLAS( cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                                    1 , 
                                                    single_input_size,
                                                    single_kernel_size, 
                                                    &alpha , 
                                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                                    1, 
                                                    im2col_buf[prev] ,
                                                    single_kernel_size , 
                                                    &beta , 
                                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                                    1
                                                ));
                                            }
                                            else if constexpr (std::is_same_v<T , double>){
                                                CHECK_CUBLAS( cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                                    1 , 
                                                    single_input_size,
                                                    single_kernel_size, 
                                                    &alpha , 
                                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                                    1, 
                                                    im2col_buf[prev] ,
                                                    single_kernel_size , 
                                                    &beta , 
                                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                                    1
                                                ));
                                            }
                                        }
                                        iter++;
                                        prev_kerneloutoffset = kerneloutoffset;
                                        prev_kernelinoffset = kernelinoffset;
                                        prev_resultoutoffset = resultoutoffset;
                                        prev_resultbatchoffset = resultbatchoffset;
                                    }
                                }
                            }

                            cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                            
                            if constexpr (std::is_same_v<T , float>){
                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                    1 , 
                                    single_input_size,
                                    single_kernel_size, 
                                    &alpha , 
                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                    1, 
                                    im2col_buf[(iter + 1) & 1] ,
                                    single_kernel_size , 
                                    &beta , 
                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                    1
                                ));
                            }
                            else if constexpr (std::is_same_v<T , double>){
                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                    1 , 
                                    single_input_size,
                                    single_kernel_size, 
                                    &alpha , 
                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                    1, 
                                    im2col_buf[(iter + 1) & 1] ,
                                    single_kernel_size , 
                                    &beta , 
                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                    1
                                ));
                            }
                        }
                        else{

                            for(inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                                for(resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                    ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                    for(inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        Functional::im2col_gpu<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>
                                        (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                        single_kernel_size , single_input_shape.size()
                                        ,kershape.get() , imshape.get() ,single_input_size);
                                        CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                        
                                        
                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                            if constexpr (std::is_same_v<T , float>){
                                                CHECK_CUBLAS( cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                                    1 , 
                                                    single_input_size,
                                                    single_kernel_size, 
                                                    &alpha , 
                                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                                    1, 
                                                    im2col_buf[prev] ,
                                                    single_kernel_size , 
                                                    &beta , 
                                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                                    1
                                                ));

                                            }
                                            else if constexpr (std::is_same_v<T ,double>){
                                                CHECK_CUBLAS( cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                                    1 , 
                                                    single_input_size,
                                                    single_kernel_size, 
                                                    &alpha , 
                                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                                    1, 
                                                    im2col_buf[prev] ,
                                                    single_kernel_size , 
                                                    &beta , 
                                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                                    1
                                                ));
                                            }
                                        }
                                        iter++;
                                        prev_kerneloutoffset = kerneloutoffset;
                                        prev_kernelinoffset = kernelinoffset;
                                        prev_resultoutoffset = resultoutoffset;
                                        prev_resultbatchoffset = resultbatchoffset;
                                    }
                                }
                            }

                            cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                            
                            if constexpr (std::is_same_v<T , float>){
                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                    1 , 
                                    single_input_size,
                                    single_kernel_size, 
                                    &alpha , 
                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                    1, 
                                    im2col_buf[(iter + 1) & 1] ,
                                    single_kernel_size , 
                                    &beta , 
                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                    1
                                ));
                                result.print();
                            }
                            else if constexpr (std::is_same_v<T , double>){
                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                    1 , 
                                    single_input_size,
                                    single_kernel_size, 
                                    &alpha , 
                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                    1, 
                                    im2col_buf[(iter + 1) & 1] ,
                                    single_kernel_size , 
                                    &beta , 
                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                    1
                                ));
                            }

                        }
                    }
                    else{

                        size_t reduce_imsize = 1;

                        for(size_t i = 2;i<resultshape.size();i++){
                            reduce_imsize *= resultshape[i];
                        }
                        if(single_kernel_shape.size() == 2){
                            for(inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                                for(resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                    ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                    for(inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        Functional::im2col_gpu_nopadding_2d<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>
                                        (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                        single_kernel_size 
                                        ,kershape.get() , imshape.get() ,single_input_size , reduce_imsize);
                                        CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                        
                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                            if constexpr (std::is_same_v<T , float>){
                                                CHECK_CUBLAS( cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                                    1 , 
                                                    reduce_imsize,
                                                    single_kernel_size, 
                                                    &alpha , 
                                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                                    1, 
                                                    im2col_buf[prev] ,
                                                    single_kernel_size , 
                                                    &beta , 
                                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                                    1
                                                ));
                                            }
                                            else if constexpr (std::is_same_v<T , double>){
                                                CHECK_CUBLAS( cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                                    1 , 
                                                    reduce_imsize,
                                                    single_kernel_size, 
                                                    &alpha , 
                                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                                    1, 
                                                    im2col_buf[prev] ,
                                                    single_kernel_size , 
                                                    &beta , 
                                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                                    1
                                                ));
                                            }
                                        }
                                        iter++;
                                        prev_kerneloutoffset = kerneloutoffset;
                                        prev_kernelinoffset = kernelinoffset;
                                        prev_resultoutoffset = resultoutoffset;
                                        prev_resultbatchoffset = resultbatchoffset;
                                    }
                                }
                            }

                            cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                            if constexpr (std::is_same_v<T , float>){

                                CHECK_CUBLAS( cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                    1 , 
                                    reduce_imsize,
                                    single_kernel_size, 
                                    &alpha , 
                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                    1, 
                                    im2col_buf[(iter + 1) & 1] ,
                                    single_kernel_size , 
                                    &beta , 
                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                    1
                                ));
                            }
                            else if constexpr (std::is_same_v<T , double>){

                                CHECK_CUBLAS( cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                    1 , 
                                    reduce_imsize,
                                    single_kernel_size, 
                                    &alpha , 
                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                    1, 
                                    im2col_buf[(iter + 1) & 1] ,
                                    single_kernel_size , 
                                    &beta , 
                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                    1
                                ));
                            }

                        }
                        else{
                            for(inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                                for(resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                    ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                    for(inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        Functional::im2col_gpu_nopadding<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>
                                        (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                        single_kernel_size , single_input_shape.size()
                                        ,kershape.get() , imshape.get() ,single_input_size , reduce_imsize);
                                        CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                        
                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                            if constexpr (std::is_same_v<T , float>){
                                                CHECK_CUBLAS( cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                                    1 , 
                                                    reduce_imsize,
                                                    single_kernel_size, 
                                                    &alpha , 
                                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                                    1, 
                                                    im2col_buf[prev] ,
                                                    single_kernel_size , 
                                                    &beta , 
                                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                                    1
                                                ));
                                            }
                                            else if constexpr (std::is_same_v<T , double>){
                                                CHECK_CUBLAS( cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                                    1 , 
                                                    reduce_imsize,
                                                    single_kernel_size, 
                                                    &alpha , 
                                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                                    1, 
                                                    im2col_buf[prev] ,
                                                    single_kernel_size , 
                                                    &beta , 
                                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                                    1
                                                ));
                                            }
                                        }
                                        iter++;
                                        prev_kerneloutoffset = kerneloutoffset;
                                        prev_kernelinoffset = kernelinoffset;
                                        prev_resultoutoffset = resultoutoffset;
                                        prev_resultbatchoffset = resultbatchoffset;
                                    }
                                }
                            }

                            cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                            if constexpr (std::is_same_v<T , float>){
                                CHECK_CUBLAS( cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                    1 , 
                                    reduce_imsize,
                                    single_kernel_size, 
                                    &alpha , 
                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                    1, 
                                    im2col_buf[(iter + 1) & 1] ,
                                    single_kernel_size , 
                                    &beta , 
                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                    1
                                ));
                            }
                            else if constexpr (std::is_same_v<T , double>){
                                CHECK_CUBLAS( cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                    1 , 
                                    reduce_imsize,
                                    single_kernel_size, 
                                    &alpha , 
                                    kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                    1, 
                                    im2col_buf[(iter + 1) & 1] ,
                                    single_kernel_size , 
                                    &beta , 
                                    result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                    1
                                ));
                            }

                        }
                    }
                }
                else{
                    std::vector<size_t> single_output_stride(resultstride.begin() , resultstride.end() - 2);
                    for(size_t inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                        ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                            for(size_t resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                    for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                        Tensor<T> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                        Functional::im2col_ptr<T>(inputtlide.get() , input.get() + inputbatchoffset + inputinoffset
                                        ,single_input_shape , single_kernel_shape ,  single_input_stride , single_output_stride , input.device());
                                        for(size_t i = 0;i< inputtlide.shape()[0];i++){
                                            T sum = 0;
                                            for(size_t k = 0;k<inputtlide.shape()[1];k++){
                                                sum += inputtlide.get()[i * inputtlide.shape()[1] + k] * 
                                                kernel_.get()[kerneloutoffset+kernelinoffset+ k];
                                            }
                                            result.get()[resultbatchoffset + resultoutoffset + i] += sum;
                                        }
                                    }
                                }
                    }

                }
                if(input.requires_grad()){
                    result.set_requires_grad(true);
                    result.set_grad_fn(
                        std::make_shared<Functional::ModuleFunctionWrapper<T> >(this , input  ));
                }
                return result;
                    
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                // grad_out (N , C_out , ...) kernel (C_out , C_in , ...) input (N , C_in , ...)
                auto input = input_cache;
                // first calculating kernel 
                auto inputstride = input.get_strides();
                auto kernelstride = kernel_.get_strides();
                std::vector<size_t> outstride = grad_out.get_strides();
                std::vector<size_t> single_kernel_shape(kernel_.shape().begin()+2 , kernel_.shape().end());
                std::vector<size_t> single_input_shape(input.shape().begin()+2 , input.shape().end());
                std::vector<size_t> single_input_stride(inputstride.begin() , inputstride.end() - 2);
                std::vector<size_t> single_output_stride(outstride.begin() , outstride.end() - 2);
                Tensor<T>  grad_kernel(kernel_.shape() , kernel_.device());
                Tensor<T>  grad_input(input.shape() , input.device());
                if(grad_kernel.device() == Cuda){

                    auto single_kernel_size = Functional::prod_vec(single_kernel_shape);
                    size_t single_input_size = Functional::prod_vec(single_input_shape);
                    size_t im2col_size = single_input_size * single_kernel_size * sizeof(T);

                    if(im2col_buf[0] == nullptr){
                        CHECK(cudaMalloc(&im2col_buf[0] , im2col_size));
                        CHECK(cudaMalloc(&im2col_buf[1] , im2col_size));
                        im2col_size_prev = im2col_size;
                    }
                    else if(im2col_size_prev != im2col_size){
                        CHECK(cudaFree(im2col_buf[0]));
                        CHECK(cudaFree(im2col_buf[1]));
                        CHECK(cudaMalloc(&im2col_buf[0] , im2col_size));
                        CHECK(cudaMalloc(&im2col_buf[1] , im2col_size));
                        im2col_size_prev = im2col_size;
                    }

                    T alpha = 1.0f;
                    T beta = 1.0f;
                    T beta0 = 0;

                    size_t iter = 0;

                    //preprocessing
                    auto instride = single_input_stride;
                    auto revinstride = instride;
                    std::reverse(revinstride.begin() , revinstride.end());
                    for(int i = 0;i<half_kernel_shape.size();i++){
                        half_kernel_shape[i] /= 2;
                    }
                    cuda_shared_pointer<size_t> kershape(single_kernel_shape , Cuda);
                    cuda_shared_pointer<size_t> imshape(single_input_shape ,Cuda);
                    size_t prev_gradoutoffset = 0 , prev_gradbatchoffset = 0 , prev_kerneloutoffset = 0 , prev_kernelinoffset = 0;

                    if(padding_mode == ZeroPadding) {
                        if(single_kernel_shape.size() == 2){
                            for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                        ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;


                                        Functional::im2col_gpu_2d_t<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>
                                        (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                        single_kernel_size 
                                        ,kershape.get() , imshape.get() ,single_input_size);
                                        CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                        
                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                            if constexpr (std::is_same_v<T , float>){
                                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                                    , 1 , single_kernel_size , single_input_size
                                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                                    im2col_buf[prev] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                            }
                                            else if constexpr (std::is_same_v<T , double>){
                                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                                    , 1 , single_kernel_size , single_input_size
                                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                                    im2col_buf[prev] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                            }
                                        }
                                        iter++;
                                        prev_kerneloutoffset = kerneloutoffset;
                                        prev_kernelinoffset = kernelinoffset;
                                        prev_gradoutoffset = gradoutoffset;
                                        prev_gradbatchoffset = gradbatchoffset;
                                    }
                                }
                            }
                            cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                            if constexpr (std::is_same_v<T , float>){
                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                    , 1 , single_kernel_size , single_input_size
                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                    im2col_buf[(iter + 1) & 1] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                            }
                            else if constexpr (std::is_same_v<T ,double>){
                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                    , 1 , single_kernel_size , single_input_size
                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                    im2col_buf[(iter + 1) & 1] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                            }
                            kernel_.set_grad(grad_kernel);
                            size_t prev_inputbatchoffset = 0 , prev_inputinoffset = 0;
                            iter = 0;

                            for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                    ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){

                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        if constexpr (std::is_same_v<T , float>){
                                            CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                                ,single_kernel_size , single_input_size , 1
                                                , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                                grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta0 ,
                                                im2col_buf[cur] , single_kernel_size
                                            ));
                                        }
                                        else if constexpr (std::is_same_v <T , double>){
                                            CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                                ,single_kernel_size , single_input_size , 1
                                                , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                                grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta0 ,
                                                im2col_buf[cur] , single_kernel_size
                                            ));

                                        }
                                        CHECK(cudaEventRecord(ev[cur] , stream_gemm));

                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_im2col , ev[prev] , 0);
                                            Functional::col2im_gpu_2d<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                                            , im2col_buf[(iter + 1) & 1] , single_kernel_size 
                                            ,kershape.get() , imshape.get() , single_input_size);
                                        }

                                        iter++;
                                        prev_inputbatchoffset = inputbatchoffset;
                                        prev_inputinoffset = inputinoffset;
                                    }
                                }
                            }

                            cudaStreamWaitEvent(stream_im2col , ev[(iter + 1) & 1] , 0);
                            Functional::col2im_gpu<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                            , im2col_buf[(iter + 1) & 1] , single_kernel_size , single_input_shape.size()
                            ,kershape.get() , imshape.get() , single_input_size);

                        }
                        else{
                            for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                        ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        Functional::im2col_gpu_t<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>
                                        (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                        single_kernel_size , single_input_shape.size()
                                        ,kershape.get() , imshape.get() ,single_input_size);
                                        CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                        
                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                            if constexpr (std::is_same_v<T , float>){
                                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                                    , 1 , single_kernel_size , single_input_size
                                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                                    im2col_buf[prev] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                            }
                                            else if constexpr (std::is_same_v<T , double>){
                                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                                    , 1 , single_kernel_size , single_input_size
                                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                                    im2col_buf[prev] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                            }
                                        }
                                        iter++;
                                        prev_kerneloutoffset = kerneloutoffset;
                                        prev_kernelinoffset = kernelinoffset;
                                        prev_gradoutoffset = gradoutoffset;
                                        prev_gradbatchoffset = gradbatchoffset;
                                    }
                                }
                            }
                            cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                            if constexpr (std::is_same_v<T , float>){
                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                    , 1 , single_kernel_size , single_input_size
                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                    im2col_buf[(iter + 1) & 1] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                            }
                            else if constexpr (std::is_same_v<T , double>){
                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                    , 1 , single_kernel_size , single_input_size
                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                    im2col_buf[(iter + 1) & 1] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));

                            }
                            kernel_.set_grad(grad_kernel);
                            size_t prev_inputbatchoffset = 0 , prev_inputinoffset = 0;
                            iter = 0;

                            for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                    ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){

                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        if constexpr (std::is_same_v<T , float>){
                                            CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                                ,single_kernel_size , single_input_size , 1
                                                , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                                grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta0 ,
                                                im2col_buf[cur] , single_kernel_size
                                            ));
                                        }
                                        else if constexpr (std::is_same_v<T , double>){
                                            CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                                ,single_kernel_size , single_input_size , 1
                                                , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                                grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta0 ,
                                                im2col_buf[cur] , single_kernel_size
                                            ));
                                        }
                                        CHECK(cudaEventRecord(ev[cur] , stream_gemm));

                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_im2col , ev[prev] , 0);
                                            Functional::col2im_gpu<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                                            , im2col_buf[(iter + 1) & 1] , single_kernel_size , single_input_shape.size()
                                            ,kershape.get() , imshape.get() , single_input_size);
                                        }

                                        iter++;
                                        prev_inputbatchoffset = inputbatchoffset;
                                        prev_inputinoffset = inputinoffset;
                                    }
                                }
                            }

                            cudaStreamWaitEvent(stream_im2col , ev[(iter + 1) & 1] , 0);
                            Functional::col2im_gpu<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                            , im2col_buf[(iter + 1) & 1] , single_kernel_size , single_input_shape.size()
                            ,kershape.get() , imshape.get() , single_input_size);

                        }

                    }
                    else{
                        size_t reduce_imsize = 1;

                        for(size_t i = 2;i<grad_out.shape().size();i++){
                            reduce_imsize *= grad_out.shape()[i];
                        }
                        if(single_kernel_shape.size() == 2){
                            for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                        ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        Functional::im2col_gpu_nopadding_2d_t<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>
                                        (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                        single_kernel_size 
                                        ,kershape.get() , imshape.get() ,single_input_size , reduce_imsize);
                                        CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                        
                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                            if constexpr (std::is_same_v<T , float>){
                                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                                    , 1 , single_kernel_size , reduce_imsize
                                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                                    im2col_buf[prev] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                            }
                                            else if constexpr (std::is_same_v<T , double>){
                                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                                    , 1 , single_kernel_size , reduce_imsize
                                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                                    im2col_buf[prev] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                            }
                                        }
                                        iter++;
                                        prev_kerneloutoffset = kerneloutoffset;
                                        prev_kernelinoffset = kernelinoffset;
                                        prev_gradoutoffset = gradoutoffset;
                                        prev_gradbatchoffset = gradbatchoffset;
                                    }
                                }
                            }
                            cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                            if constexpr (std::is_same_v<T , float>){
                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                    , 1 , single_kernel_size , reduce_imsize
                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                    im2col_buf[(iter + 1) & 1] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                            }
                            else if constexpr (std::is_same_v<T , double>){
                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                    , 1 , single_kernel_size , reduce_imsize
                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                    im2col_buf[(iter + 1) & 1] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                            }
                            kernel_.set_grad(grad_kernel);
                            size_t prev_inputbatchoffset = 0 , prev_inputinoffset = 0;
                            iter = 0;

                            for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                    ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){

                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        if constexpr (std::is_same_v<T , float>){
                                            CHECK_CUBLAS(
                                            cublasSgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                                ,single_kernel_size , reduce_imsize , 1
                                                , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                                grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta0 ,
                                                im2col_buf[cur] , single_kernel_size
                                            ));
                                        }
                                        else if constexpr (std::is_same_v<T , double>){
                                            CHECK_CUBLAS(
                                            cublasDgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                                ,single_kernel_size , reduce_imsize , 1
                                                , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                                grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta0 ,
                                                im2col_buf[cur] , single_kernel_size
                                            ));
                                        }
                                        CHECK(cudaEventRecord(ev[cur] , stream_gemm));

                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_im2col , ev[prev] , 0);
                                            Functional::col2im_gpu_nopadding_2d<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                                            , im2col_buf[prev] , single_kernel_size 
                                            ,kershape.get() , imshape.get() , single_input_size , reduce_imsize);
                                        }

                                        iter++;
                                        prev_inputbatchoffset = inputbatchoffset;
                                        prev_inputinoffset = inputinoffset;
                                    }
                                }
                            }

                            cudaStreamWaitEvent(stream_im2col , ev[(iter + 1) & 1] , 0);
                            Functional::col2im_gpu_nopadding_2d<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                            , im2col_buf[(iter + 1) & 1] , single_kernel_size 
                            ,kershape.get() , imshape.get() , single_input_size , reduce_imsize);

                        }
                        else{
                            for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                        ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        Functional::im2col_gpu_nopadding_t<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>
                                        (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                        single_kernel_size , single_input_shape.size()
                                        ,kershape.get() , imshape.get() ,single_input_size , reduce_imsize);
                                        CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                        
                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                            if constexpr (std::is_same_v<T , float>){
                                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                                    , 1 , single_kernel_size , reduce_imsize
                                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                                    im2col_buf[prev] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                            }
                                            else if constexpr (std::is_same_v<T , double>){
                                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                                    , 1 , single_kernel_size , reduce_imsize
                                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                                    im2col_buf[prev] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                            }
                                        }
                                        iter++;
                                        prev_kerneloutoffset = kerneloutoffset;
                                        prev_kernelinoffset = kernelinoffset;
                                        prev_gradoutoffset = gradoutoffset;
                                        prev_gradbatchoffset = gradbatchoffset;
                                    }
                                }
                            }
                            cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                            if constexpr (std::is_same_v<T , float>){
                                CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                    , 1 , single_kernel_size , reduce_imsize
                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                    im2col_buf[(iter + 1) & 1] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                            }
                            else if constexpr (std::is_same_v<T , double>){
                                CHECK_CUBLAS(cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                    , 1 , single_kernel_size , reduce_imsize
                                    , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                    im2col_buf[(iter + 1) & 1] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                            }
                            kernel_.set_grad(grad_kernel);
                            size_t prev_inputbatchoffset = 0 , prev_inputinoffset = 0;
                            iter = 0;

                            for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                    ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){

                                        size_t cur = iter & 1;
                                        size_t prev = (iter + 1) & 1;

                                        if constexpr (std::is_same_v<T , float>){
                                            cublasSgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                                ,single_kernel_size , reduce_imsize , 1
                                                , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                                grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta0 ,
                                                im2col_buf[cur] , single_kernel_size
                                            );
                                        }
                                        else if constexpr (std::is_same_v<T , double>){
                                            cublasDgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                                ,single_kernel_size , reduce_imsize , 1
                                                , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                                grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta0 ,
                                                im2col_buf[cur] , single_kernel_size
                                            );
                                        }
                                        CHECK(cudaEventRecord(ev[cur] , stream_gemm));

                                        if(iter > 0){
                                            cudaStreamWaitEvent(stream_im2col , ev[prev] , 0);
                                            Functional::col2im_gpu_nopadding<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                                            , im2col_buf[prev] , single_kernel_size , single_input_shape.size()
                                            ,kershape.get() , imshape.get() , single_input_size , reduce_imsize);
                                        }

                                        iter++;
                                        prev_inputbatchoffset = inputbatchoffset;
                                        prev_inputinoffset = inputinoffset;
                                    }
                                }
                            }

                            cudaStreamWaitEvent(stream_im2col , ev[(iter + 1) & 1] , 0);
                            Functional::col2im_gpu_nopadding<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                            , im2col_buf[(iter + 1) & 1] , single_kernel_size , single_input_shape.size()
                            ,kershape.get() , imshape.get() , single_input_size , reduce_imsize);

                        }

                    }

                }
                else{
                    for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                        ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                        for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                        ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                            for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                    Tensor<T> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                    Functional::im2col_ptr<T>(inputtlide.get() , input.get() + inputbatchoffset + inputinoffset
                                        ,single_input_shape , single_kernel_shape ,  single_input_stride , single_output_stride , input.device());
                                    for(size_t i = 0;i< inputtlide.shape()[1];i++){
                                        T sum = 0;
                                        for(size_t k = 0;k<inputtlide.shape()[0];k++){
                                            sum += inputtlide.get()[i + inputtlide.shape()[1] * k] * 
                                            grad_out.get()[gradbatchoffset + gradoutoffset + k];
                                        }
                                        grad_kernel.get()[kerneloutoffset + kernelinoffset + i] += sum;
                                    }
                            }
                        }
                    }

                    kernel_.set_grad(grad_kernel);
                    for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                        ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                        for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                        ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                            for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                            ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                                Tensor<T> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                    for(size_t i = 0;i<inputtlide.shape()[0];i++){
                                        for(size_t j = 0;j < inputtlide.shape()[1];j++){
                                            inputtlide.get()[i * inputtlide.shape()[1] + j] += 
                                            grad_out.get()[gradbatchoffset + gradoutoffset + i] * 
                                            kernel_.get()[kerneloutoffset + kernelinoffset + j];
                                        }
                                    }
                                Functional::col2im_ptr(grad_input.get() + inputbatchoffset + inputinoffset , inputtlide.get()
                                , single_input_shape , single_kernel_shape , single_input_stride , single_output_stride , input.device());
                            }
                        }
                    }

                }
                return {grad_input};
            }
            std::vector<Tensor<T>> parameters() override{
                return {kernel_};
            }
    };


    template <typename T>
    class Pool2d : public Module<T>{
        private:
            std::vector<size_t> kernel_shape_;
            Tensor<T> input_cache;
            std::shared_ptr<Functional::Pool2dFunc<T>> pool2d_func;
        public:
            Pool2d(std::vector<size_t> kernel_shape) : kernel_shape_(kernel_shape){
                pool2d_func = std::make_shared<Functional::Pool2dFunc<T>>(kernel_shape_);
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & input) override{
                if(input[0].requires_grad()){
                    Tensor<T> result = pool2d_func->forward(input);
                    result.set_grad_fn(pool2d_func);
                    input_cache = input[0];
                    return result;
                }
                return Functional::Pool2dFunc<T>(kernel_shape_).forward(input);
            }
            std::vector<Tensor<T>> parameters() override{
                return {};
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                return pool2d_func->backward(grad_out);
            }
    };

    

    
    __device__ float warpReduceMax(float val);
    __device__ float warpReduceSum(float val);
    __device__ double warpReduceMax_double(double val);
    __device__ double warpReduceSum_double(double val);
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
                size_t stride = input.shape().back();
                size_t batchsize = input.size() / stride;
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
                    for(size_t i = 0;i < input.size() / stride;i++){
                        T sum = 0;
                        for(size_t j = 0;j < stride;j++){
                            output.get()[i * stride + j] = exp(input.get()[i * stride + j]);
                            sum += output.get()[i * stride + j];
                        }
                        for(size_t j = 0;j < stride;j++){
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
    __global__ void _cross_entropy_backward_kernel(T * grad_input , const T * input_softmax , const T * grad_out , const size_t * label_cache , const size_t batchsize , 
    const size_t step){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= batchsize * step){
            return;
        }
        grad_input[index] = (input_softmax[index] - (label_cache[index / step] == index % step)) * grad_out[0] / batchsize;
    }

    template <typename T>
    __global__ void _cross_entropy_forward_kernel(T * loss , const T * input_softmax , const size_t * label_cache , const size_t batchsize , const size_t step){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        extern __shared__ char smem[];
        T * smem_ce = reinterpret_cast<T *>(smem);
        int warpId = threadIdx.x / 32;
        int laneId = threadIdx.x % 32;
        int warpsPerBlock = blockDim.x / 32;
        T l;
        if constexpr (std::is_same_v<T , float>){
            l = index < batchsize ? -logf(input_softmax[index * step + label_cache[index]]) : 0;
        }
        else if constexpr (std::is_same_v<T , double>){
            l = index < batchsize ? -log(input_softmax[index * step + label_cache[index]]) : 0;
        }
        l = warpReduceSum(l); // sum in warp
        if(laneId == 0){
            smem_ce[warpId] = l;
        }
        __syncthreads();
        if(threadIdx.x == 0){
            float val = smem_ce[0];
            for(int i = 1;i < warpsPerBlock;i++){
                val += smem_ce[i];
            }
            atomicAdd(loss , val);
        }
    }
    template <typename T>
    __global__ void divideSingleElement(T * loss , const T batchsize){
        loss[0] /= batchsize;
    }


    template <typename T>
    class CrossEntropy : public Module<T>{
        private:
            Softmax<T> softmax;
            Tensor<T> input_cache , input_softmax_cache;
            cuda_shared_pointer<size_t> label_cache_;
        public:
            CrossEntropy(const std::vector<size_t> & label_cache) {
                label_cache_ = cuda_shared_pointer<size_t>(label_cache , DefaultDevice);
            }
            CrossEntropy(const Tensor<T> & label_cache){
                auto label_cache_tensor = label_cache.deepcopy();
                label_cache_ = label_cache_tensor.get_shared_ptr();
            }
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                if(inputs.size()!= 1){
                    std::cerr << "CrossEntropy input size must be 1" << std::endl;
                    throw std::runtime_error("CrossEntropy input size must be 1");
                }
                auto input = inputs[0];
                auto input_softmax = softmax(input);
                if(input.requires_grad()){
                    input_cache = input;
                    input_softmax_cache = input_softmax;
                }
                Tensor<T> loss(T(0) , {1} , input.device());
                size_t batchsize = input.size() / input.shape().back();
                size_t step = input.shape().back();
                if(input.device() == Cpu){
                    for(size_t i = 0;i < batchsize;i++){
                        loss.get()[0] += -log(input_softmax.get()[i * step + label_cache_.get()[i]]);
                    }
                    loss.get()[0] /= batchsize;
                }
                else{
                    _cross_entropy_forward_kernel<<<CudaGetBlocks(batchsize) , kCudaThreadsNum>>>(
                        loss.get() , input_softmax.get() , label_cache_.get() , batchsize , step
                    );
                    divideSingleElement<T><<<1 , 1>>>(loss.get() , batchsize);
                }
                if(input.requires_grad()){
                    loss.set_requires_grad(true);
                    loss.set_grad_fn(
                        std::make_shared<Functional::ModuleFunctionWrapper<float> >(this , input  ));
                }
                return loss;
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                size_t batchsize = input_cache.size() / input_cache.shape().back();
                size_t step = input_cache.shape().back();
                Tensor<T> grad_input(input_cache.shape() , input_cache.device());
                if(input_cache.device() == Cpu){
                    for(int i = 0;i<batchsize;i++){
                        for(int j = 0;j<step;j++){
                            grad_input.get()[i * step + j] = (input_softmax_cache.get()[i*step + j] - (j == label_cache_[i])) * grad_out.get()[0] / batchsize;
                        }
                    }
                }
                else{
                    input_softmax_cache.to(Cuda);
                    cuda_shared_pointer<size_t> label_cache_cuda(label_cache_ , Cuda);
                    _cross_entropy_backward_kernel<T><<<CudaGetBlocks(input_cache.size()) , kCudaThreadsNum>>>(
                        grad_input.get() , input_softmax_cache.get() , grad_out.get() , label_cache_cuda.get() , batchsize , step
                    );
                }
                return {grad_input};
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
    __global__ void _batch_norm_kernel1(
        T * out_mean, T * out_var,
        const T * input , const size_t hw , const size_t c , const size_t batch_size , const size_t size ,const T momentum){
        extern __shared__ char smem[];
        T * bn_smem = reinterpret_cast<T *>(smem);
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int warpId = tid / 32;
        int laneId = tid % 32;
        constexpr int warpsPerBlock =   kCudaThreadsNum / 32;
        T sum = 0;
        int chw = c * hw;
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                sum += input[i + offset];
            }
        }
        sum = warpReduceSum(sum);
        T hw_batch_size = hw * batch_size;
        __syncthreads();
        if(laneId == 0) bn_smem[warpId] = sum / hw_batch_size;
        __syncthreads();
        if(tid == 0){
            T val = bn_smem[0];
            for(int i = 1;i < warpsPerBlock;i++){
                val += bn_smem[i];
            }
            out_mean[cid] = val * momentum + out_mean[cid] * (1 - momentum);
        }
        __syncthreads();
        sum = 0;
        const T mean = out_mean[cid];
        
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                T diff = input[i + offset] - mean;
                sum += diff * diff;
            }
        }
        sum = warpReduceSum(sum);
        __syncthreads();
        if(laneId == 0) bn_smem[warpId] = sum / hw_batch_size;
        __syncthreads();
        if(tid == 0){
            T val = bn_smem[0];
            for(int i = 1;i < warpsPerBlock;i++){
                val += bn_smem[i];
            }
            out_var[cid] = val * momentum + out_var[cid] * (1 - momentum);
        }

    }
    template <typename T>
    __global__ void _batch_norm_kernel2(
        T * output , const T * input , const T * running_mean , const T * running_var , const T * gamma , const T * beta , const T epsilon,
        const size_t hw , const size_t c , const size_t batch_size , const size_t size
    ){
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int chw = c * hw;
        T mean = running_mean[cid];
        T var = running_var[cid];
        T var_inv = rsqrt(var + epsilon);
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
    __global__ void _batch_norm_kernel_beta(
        T * grad_in,
        const T * grad_out , const size_t hw , const size_t c , const size_t size){
        extern __shared__ char smem[];
        T * bn_smem = reinterpret_cast<T *>(smem);
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int warpId = tid / 32;
        int laneId = tid % 32;
        constexpr int warpsPerBlock =   kCudaThreadsNum / 32;
        T sum = 0;
        int chw = c * hw;
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                sum += grad_out[i + offset];
            }
        }
        sum = warpReduceSum(sum);
        __syncthreads();
        if(laneId == 0) bn_smem[warpId] = sum;
        __syncthreads();
        if(tid == 0){
            T val = bn_smem[0];
            for(int i = 1;i < warpsPerBlock;i++){
                val += bn_smem[i];
            }
            grad_in[cid] = val;
        }

    }
    template <typename T>
    __global__ void _batch_norm_kernel_gamma(
        T * grad_gamma , const T * grad_out , const T * input , const T * running_mean , const T * running_var , const T * gamma  , const T epsilon,
        const size_t hw , const size_t c  , const size_t size
    ){
        extern __shared__ char smem[];
        T * bn_smem = reinterpret_cast<T *>(smem);
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int warpId = tid / 32;
        int laneId = tid % 32;
        constexpr int warpsPerBlock =   kCudaThreadsNum / 32;
        T sum = 0;
        int chw = c * hw;
        T mean = running_mean[cid];
        T var_inv = rsqrt(running_var[cid] + epsilon);
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                sum += grad_out[i + offset] * (input[i + offset] - mean) * var_inv;
            }
        }
        sum = warpReduceSum(sum);
        __syncthreads();
        if(laneId == 0) bn_smem[warpId] = sum;
        __syncthreads();
        if(tid == 0){
            T val = bn_smem[0];
            for(int i = 1;i < warpsPerBlock;i++){
                val += bn_smem[i];
            }
            grad_gamma[cid] = val;
        }
    }

    template <typename T>
    __global__ void _batch_norm_kernel_in(
        T * grad_in , const T * grad_out , const T * running_var , const T * gamma, const T epsilon,
        const int hw , const int c , const int size
    ){
        int tid = threadIdx.x;
        int cid = blockIdx.x;
        int chw = c * hw;
        T var = running_var[cid];
        T var_inv = rsqrt(var + epsilon);
        T gamma_cid = gamma[cid];
        for(int offset = cid * hw;offset < size; offset += chw){
            for(int i = tid;i < hw;i += kCudaThreadsNum){
                grad_in[i + offset ] = grad_out[i + offset] * gamma_cid * var_inv;
            }
        }
    }



    template <typename T>
    class BatchNorm2d : public Module<T>{
        size_t c;
        Tensor<T> gamma , beta;
        Tensor<T> running_mean , running_var;
        Tensor<T> input_cache;
        T momentum;
        public:
             BatchNorm2d(const size_t num_features , T momentum = 0.1 , Device device = DefaultDevice) : c(num_features) , momentum(momentum){
                gamma = ones<T>({num_features} , device);
                beta = zeros<T>({num_features} , device);
                running_mean = zeros<T>({num_features} , device);
                running_var = ones<T>({num_features} , device);
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
                if(this->training){
                    int n = inputshape[0];
                    int h = inputshape[2];
                    int w = inputshape[3];
                    int hw = h * w;
                    int chw = c * hw;
                    int size = chw * n;
                    int hw_batch_size = hw * n;
                    if(input.device() == Cuda){
                        _batch_norm_kernel1<T><<<c , kCudaThreadsNum , sizeof(T) * (kCudaThreadsNum / 32)>>>(
                            running_mean.get() , running_var.get() , input.get() , hw  , c , n , size , momentum
                        );
                        _batch_norm_kernel2<T><<<c , kCudaThreadsNum>>>(
                            result.get() , input.get() , running_mean.get() , running_var.get() , gamma.get() , beta.get() , 1e-8,
                            hw  , c , n , size
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
                            running_mean[cid] = mean * momentum + running_mean[cid] * (1 - momentum);
                            running_var[cid] = var * momentum + running_var[cid] * (1 - momentum);
                            T var_inv = rsqrt(running_var[cid] + 1e-8);
                            mean = running_mean[cid];
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
                            result.get() , input.get() , running_mean.get() , running_var.get() , gamma.get() , beta.get() , 1e-8,
                            hw  , c , n , size
                        );
                    }
                    else{
                        for(int cid = 0; cid < c;cid++){
                            T var_inv = rsqrt(running_var[cid] + 1e-8);
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
                return result;
            }
            std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out) override{
                auto input = input_cache;
                auto inputshape = grad_out.shape();
                Tensor<T> grad_in = Tensor<T>(inputshape , input.device());
                int n = inputshape[0];
                int h = inputshape[2];
                int w = inputshape[3];
                Tensor<T> grad_beta({c} , input.device());
                Tensor<T> grad_gamma({c} , input.device());
                int hw = h * w;
                int chw = c * hw;
                int size = chw * n;
                if(input.device() == Cuda){
                    _batch_norm_kernel_beta<T><<<c , kCudaThreadsNum , sizeof(T) * (kCudaThreadsNum / 32)>>>(
                        grad_beta.get() , grad_out.get() , hw , c , size
                    );
                    _batch_norm_kernel_gamma<T><<<c , kCudaThreadsNum , sizeof(T) * (kCudaThreadsNum / 32)>>>(
                        grad_gamma.get() , grad_out.get() , input.get() , running_mean.get() , running_var.get() , gamma.get() , 1e-8,
                        hw , c , size
                    );
                    _batch_norm_kernel_in<T> <<< c , kCudaThreadsNum>>>(
                        grad_in.get() , grad_out.get() , running_var.get() , gamma.get() , 1e-8,
                        hw , c , size
                    );
                }
                else{
                    for(int cid = 0; cid < c;cid++){
                        T sum_grad = 0;
                        T sum_grad_gamma = 0;
                        T mean = running_mean[cid];
                        T var_inv = rsqrt(running_var[cid] + 1e-8);
                        for(int offset = cid * hw;offset < size; offset += chw){
                            for(int i = 0;i < hw;i++){
                                sum_grad += grad_out[i + offset];
                                sum_grad_gamma += grad_out[i + offset] * (input[i + offset] - mean) * var_inv;
                            }
                        }
                        grad_beta[cid] = sum_grad;
                        grad_gamma[cid] = sum_grad_gamma;
                        for(int offset = cid * hw;offset < size; offset += chw){
                            for(int i = 0;i < hw;i++){
                                grad_in[i + offset] = grad_out[i + offset] * gamma[cid] * var_inv;
                            }
                        }

                    }
                }
                gamma.set_grad(grad_gamma);
                beta.set_grad(grad_beta);
                return {grad_in};
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
                std::vector<size_t> newshape(shape.begin() , shape.begin() + start_dim);
                newshape.push_back(newsize);
                newshape.insert(newshape.end() , shape.begin() + end_dim + 1 , shape.end());
                return input.reshape(newshape);
            }
    };

}
}


#endif