#ifndef _NN_H_
#define _NN_H_

#include "tensor.cuh"
#include <cublas_v2.h>

namespace mytorch{
namespace nn{
    constexpr size_t kCudaTransposeTileSize = 4;
    constexpr size_t kCudaMultiDimMax = 16;
    template <typename T>
    class Module{
        public:
            
            Module() = default;
            virtual Tensor<T> forward(const std::vector<Tensor<T>> & input) = 0;
            Tensor<T> operator()(const std::vector<Tensor<T>> & input){
                return forward(input);
            }
            Tensor<T> operator()(const Tensor<T> & input){
                return forward({input});
            }
            virtual std::vector<Tensor<T>> _internal_backward(const Tensor<T> & grad_out){
                return {}; // default no backward
            }
            virtual std::vector<Tensor<T>> parameters() = 0;
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
                virtual Tensor<T> forward(const std::vector<Tensor<T>> & inputs) = 0;
                virtual std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) = 0;
                virtual std::vector<Tensor<T>> get_inputs() const = 0;
                Tensor<T> operator()(const std::vector<Tensor<T>> & inputs){
                    return forward(inputs);
                }
                Tensor<T> operator()(const Tensor<T> & input){
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
                output[index] = 1.0 / (1.0 + std::exp(-input[index]));
            }
        }

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
        class TransposeFunc : public Function<T>{
            private:
                std::vector<size_t> perm;
                Tensor<T> a;
            public:
                TransposeFunc(const std::vector<size_t> & perm) : perm(perm){}
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
                    return {grad_out.transpose(_get_reverse_perm(perm))};
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
                    for(size_t inputoffset = 0 , outputoffset = 0;inputoffset < input.size();inputoffset += inputstep , outputoffset += resultstep){
                        if(result.device() == Cuda){
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
                        else{
                            for(size_t i = 0;i<single_input_shape[0];i++){
                                for(int j = 0;j<single_input_shape[1];j++){
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
                    for(size_t inputoffset = 0 , outputoffset = 0;inputoffset < input.size();inputoffset += inputstep , outputoffset += resultstep){
                        if(grad_in.device() == Cuda){
                            _pool_backward_kernel<T><<<CudaGetBlocks(resultstep) , kCudaThreadsNum>>>(
                                grad_in.get() + inputoffset,grad_out.get() + outputoffset,
                                mask.get() + outputoffset,
                                single_output_size
                            );
                        }
                        else{
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
                        for(int offset0 = 0 , offset1 = 0 , offsetresult = 0;offsetresult < result.size();offset0 += step0 , offset1 += step1 , offsetresult += stepresult){
                            cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                resultshape[resultshape.size() - 1] , 
                                resultshape[resultshape.size() - 2],
                                input0shape[input0shape.size() - 1] , 
                                &alpha , 
                                input[1].get() + offset1 , 
                                input1stride[1] , 
                                input[0].get() + offset0 , 
                                input0stride[1] , 
                                &beta , 
                                result.get() + offsetresult , 
                                resultstride[1]
                            );
                        }
                        cublasDestroy(handle);
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
                        for(int offset0 = 0 , offset1 = 0 , offsetresult = 0;offsetresult < result.size();offset0 += step0 , offset1 += step1 , offsetresult += stepresult){
                            cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                resultshape[resultshape.size() - 1] , 
                                resultshape[resultshape.size() - 2],
                                input0shape[input0shape.size() - 1] , 
                                &alpha , 
                                input[1].get() + offset1 , 
                                input1stride[1] , 
                                input[0].get() + offset0 , 
                                input0stride[1] , 
                                &beta , 
                                result.get() + offsetresult , 
                                resultstride[1]
                            );
                        }
                        cublasDestroy(handle);
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
                        size_t grid_offset =  0;
                        for(int i = 0;i<ndim;i++){
                            im_offset *= imshape[i];
                            im_offset += kernel_index[i];
                            grid_offset *= kernel_shape[i];
                            grid_offset += grid_index.get_index()[i];
                        }
                        size_t col_offset = index * kernel_size + grid_offset;
                        col[col_offset] =  im[im_offset];
                    }
                    grid_index.next();

                }while(!grid_index.is_zero());

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
                        size_t grid_offset =  0;
                        for(int i = 0;i<ndim;i++){
                            im_offset *= imshape[i];
                            im_offset += kernel_index[i];
                            grid_offset *= kernel_shape[i];
                            grid_offset += grid_index.get_index()[i];
                        }
                        size_t col_offset = index * kernel_size + grid_offset;
                        atomicAdd( &im[im_offset] , col[col_offset] );
                    }
                    grid_index.next();

                }while(!grid_index.is_zero());

            }
        }




    }
    template <typename T>
    class Linear;
    template <typename T>
    __global__ void _linear_add(T * output , const T * input , size_t size){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < size){
            output[index] += input[index];
        }
    }
    template <>
    class Linear<float> : public Module<float>{
        private:
            Tensor<float> weight;
            Tensor<float> bias;
            Tensor<float> input_cache; // internal backward
        public:
            Linear(const Tensor<float> & weight, const Tensor<float> & bias)
            : weight(weight), bias(bias) {}
            Tensor<float> forward(const std::vector<Tensor<float>> & inputs) override{
                auto input = inputs[0];
                if(input.requires_grad()){
                    input_cache = input;
                }
                std::vector<size_t> newshape = input.shape();
                newshape.pop_back();
                newshape.push_back(bias.size());
                Tensor<float> result(newshape , input.device());
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
                    cublasHandle_t handle;
                    cublasCreate(&handle);
                    float alpha = 1.0f;
                    float beta = 1.0f;
                    for(int  offset1 = 0 , offsetresult = 0;offsetresult < result.size(); offset1 += step1 , offsetresult += stepresult){
                        CHECK(cudaMemcpy(result.get() + offsetresult , bias.get() ,sizeof(float) * bias.size() , cudaMemcpyDeviceToDevice));
                        cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                            1 , 
                            resultshape[resultshape.size() - 1],
                            input0shape[input0shape.size() - 1] , 
                            &alpha , 
                            input.get() + offset1 , 
                            1 , 
                            weight.get() , 
                            input0stride[1] , 
                            &beta , 
                            result.get() + offsetresult , 
                            1
                        );
                    }
                    cublasDestroy(handle);
                }
                else{
                    weight.to(Cpu);
                    bias.to(Cpu);
                    for(int  offset1 = 0 , offsetresult = 0;offsetresult < result.size(); offset1 += step1 , offsetresult += stepresult){
                        for(int i = 0;i<result.shape()[result.shape().size() - 1];i++){
                                float sum = 0.0;
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
                        std::make_shared<Functional::ModuleFunctionWrapper<float> >(this , input  ));
                    result.set_requires_grad(true);
                }
                return result;
            }
            std::vector<Tensor<float>> _internal_backward(const Tensor<float> & grad_out) override{
                auto input = input_cache;
                Tensor<float> grad_input(input.shape() , input.device());
                Tensor<float> grad_weight(weight.shape() , weight.device());
                Tensor<float> grad_bias(bias.shape() , bias.device());
                size_t stepinput = input.get_strides()[1];
                size_t stepgradout = grad_out.get_strides()[1];
                if(input.device() == Cuda){
                    cublasHandle_t handle;
                    cublasCreate(&handle);
                    float alpha = 1.0f;
                    float beta = 1.0f;
                    for(size_t inputoffset = 0 , gradoutoffset = 0;gradoutoffset < grad_out.size();inputoffset += stepinput , gradoutoffset += stepgradout){
                        cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_T,
                            1 , input.shape().back() , grad_out.shape().back(),
                            &alpha , 
                            grad_out.get() + gradoutoffset,
                            1 , 
                            weight.get() , 
                            weight.shape().back(),
                            &beta,
                            grad_input.get() + inputoffset,
                            1
                        );
                        cublasSgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N,
                            input.shape().back() ,  grad_out.shape().back(), 1,
                            &alpha , 
                            input.get() + inputoffset , 
                            1 , 
                            grad_out.get() + gradoutoffset,
                            1
                            ,&beta , 
                            grad_weight.get(),
                            grad_weight.shape().back()
                        );
                        _linear_add<float><<<CudaGetBlocks(grad_bias.size()) , kCudaThreadsNum>>>(
                            grad_bias.get() , 
                            grad_out.get() + gradoutoffset , 
                            grad_bias.size()
                        );
                    }
                    cublasDestroy(handle);

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
            std::vector<Tensor<float>> parameters() override{
                return {weight , bias};
            }
    };
    template <>
    class Linear<double> : public Module<double>{
        private:
            Tensor<double> weight;
            Tensor<double> bias;
            Tensor<double> input_cache; // internal backward
        public:
            Linear(const Tensor<double> & weight, const Tensor<double> & bias)
            : weight(weight), bias(bias) {}
            Tensor<double> forward(const std::vector<Tensor<double>> & inputs) override{
                auto input = inputs[0];
                if(input.requires_grad()){
                    input_cache = input;
                }
                std::vector<size_t> newshape = weight.shape();
                newshape.pop_back();
                Tensor<double> result(newshape , input.device());
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
                    cublasHandle_t handle;
                    cublasCreate(&handle);
                    double alpha = 1.0f;
                    double beta = 1.0f;
                    for(int  offset1 = 0 , offsetresult = 0;offsetresult < result.size(); offset1 += step1 , offsetresult += stepresult){
                        CHECK(cudaMemcpy(result.get() + offsetresult , bias.get() ,sizeof(double) * bias.size() , cudaMemcpyDeviceToDevice));
                        cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                            1 , 
                            resultshape[resultshape.size() - 1],
                            input0shape[input0shape.size() - 1] , 
                            &alpha , 
                            input.get() + offset1 , 
                            1 , 
                            weight.get() , 
                            input0stride[1] , 
                            &beta , 
                            result.get() + offsetresult , 
                            1
                        );
                    }
                    cublasDestroy(handle);
                }
                else{
                    for(int  offset1 = 0 , offsetresult = 0;offsetresult < result.size(); offset1 += step1 , offsetresult += stepresult){
                        for(int i = 0;i<result.shape()[result.shape().size() - 1];i++){
                                double sum = 0.0;
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
                        std::make_shared<Functional::ModuleFunctionWrapper<double> >(this , input  ));
                    result.set_requires_grad(true);
                }
                return result;
            }
            std::vector<Tensor<double>> _internal_backward(const Tensor<double> & grad_out) override{
                Tensor<double> grad_input(input_cache.shape() , input_cache.device());
                grad_input.set_requires_grad(false);
                auto grad_bias = grad_out;
                grad_bias.set_requires_grad(false);
                auto grad_weight = grad_out.expand(grad_out.shape().size()).matmul(input_cache.expand(input_cache.shape().size()).transpose());
                grad_weight.set_requires_grad(false);
                weight.set_grad(grad_weight);
                bias.set_grad(grad_bias);
                return {weight.transpose().matmul(grad_out.expand(grad_out.shape().size())).reshape(input_cache.shape())};
            }
            std::vector<Tensor<double>> parameters() override{
                return {weight , bias};
            }
    };

    template <typename T>
    class Conv;
    template <>
    class Conv<float> : public Module<float>{
        private:
            Tensor<float> kernel_;
            Tensor<float> input_cache;
        public:
            Conv(const Tensor<float> & kernel)
            : kernel_(kernel) {}
            Tensor<float> forward(const std::vector<Tensor<float>> & inputs) override{
                auto input = inputs[0];
                // input :(N , C_in , ...) kernel (C_out , C_in , ...) result (N , C_out , ...)
                if(input.ndim() != kernel_.ndim()){
                    throw std::runtime_error("Conv: input and kernel must have the same number of dimensions");
                }
                if(input.shape()[1] != kernel_.shape()[1]){
                    throw std::runtime_error("Conv: input and kernel must have the same number of input channels");
                }
                if(input.requires_grad()){
                    input_cache = input;
                }
                kernel_.to(input.device());
                auto resultshape = input.shape();
                resultshape[1] = kernel_.shape()[0];
                Tensor<float> result(resultshape , input.device());
                std::vector<size_t> resultstride = result.get_strides();
                auto inputstride = input.get_strides();
                auto kernelstride = kernel_.get_strides();
                std::vector<size_t> single_kernel_shape(kernel_.shape().begin()+2 , kernel_.shape().end());
                std::vector<size_t> single_input_shape(input.shape().begin()+2 , input.shape().end());
                std::vector<size_t> single_input_stride(inputstride.begin() , inputstride.end() - 2);
                std::vector<size_t> single_output_stride(resultstride.begin() , resultstride.end() - 2);
                for(size_t inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                    ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                        for(size_t resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                            ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    Tensor<float> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                    Functional::im2col_ptr<float>(inputtlide.get() , input.get() + inputbatchoffset + inputinoffset
                                     ,single_input_shape , single_kernel_shape ,  single_input_stride , single_output_stride , input.device());
                                    if(result.device() == Cuda){
                                        cublasHandle_t handle;
                                        cublasCreate(&handle);
                                        float alpha = 1.0f;
                                        float beta = 1.0f;
                                        cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                            1 , 
                                            inputstride[input.ndim() - 2],
                                            kernelstride[kernelstride.size() - 2], 
                                            &alpha , 
                                            kernel_.get() + kerneloutoffset + kernelinoffset , 
                                            1, 
                                            inputtlide.get(), 
                                            inputtlide.shape().back()
                                            , 
                                            &beta , 
                                            result.get() + resultbatchoffset + resultoutoffset , 
                                            1
                                        );
                                        cublasDestroy(handle);
                                    }
                                    else{
                                        for(size_t i = 0;i< inputtlide.shape()[0];i++){
                                            float sum = 0;
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
                        std::make_shared<Functional::ModuleFunctionWrapper<float> >(this , input  ));
                }
                return result;
                    
            }
            std::vector<Tensor<float>> _internal_backward(const Tensor<float> & grad_out) override{
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
                Tensor<float>  grad_kernel(kernel_.shape() , kernel_.device());
                for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                    ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                    for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                        for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                            ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                Tensor<float> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                Functional::im2col_ptr<float>(inputtlide.get() , input.get() + inputbatchoffset + inputinoffset
                                    ,single_input_shape , single_kernel_shape ,  single_input_stride , single_output_stride , input.device());
                                if(kernel_.device() == Cuda){
                                    cublasHandle_t handle;
                                    cublasCreate(&handle);
                                    float alpha = 1.0f;
                                    float beta = 1.0f;
                                    cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_T
                                        , 1 , inputtlide.shape()[1] , inputtlide.shape()[0]
                                        , &alpha , grad_out.get() + gradbatchoffset + gradoutoffset, 1 , 
                                        inputtlide.get() , inputtlide.shape().back() , &beta , grad_kernel.get() + kerneloutoffset + kernelinoffset , 1);
                                    cublasDestroy(handle);
                                }
                                else{
                                    for(size_t i = 0;i< inputtlide.shape()[1];i++){
                                        float sum = 0;
                                        for(size_t k = 0;k<inputtlide.shape()[0];k++){
                                            sum += inputtlide.get()[i + inputtlide.shape()[1] * k] * 
                                            grad_out.get()[gradbatchoffset + gradoutoffset + k];
                                        }
                                        grad_kernel.get()[kerneloutoffset + kernelinoffset + i] += sum;
                                    }
                                }
                        }
                    }
                }
                kernel_.set_grad(grad_kernel);
                Tensor<float>  grad_input(input.shape() , input.device());
                for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                    ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                    for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                        for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                        ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                            Tensor<float> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                            if(kernel_.device() == Cuda){
                                cublasHandle_t handle;
                                cublasCreate(&handle);
                                float alpha = 1.0f;
                                float beta = 1.0f;
                                cublasSgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                    ,inputtlide.shape()[1] , inputtlide.shape()[0] , 1
                                    , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                    grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta ,
                                    inputtlide.get() , inputtlide.shape().back()
                                );
                                cublasDestroy(handle);
                            }
                            else{
                                for(size_t i = 0;i<inputtlide.shape()[0];i++){
                                    for(size_t j = 0;j < inputtlide.shape()[1];j++){
                                        inputtlide.get()[i * inputtlide.shape()[1] + j] += 
                                        grad_out.get()[gradbatchoffset + gradoutoffset + i] * 
                                        kernel_.get()[kerneloutoffset + kernelinoffset + j];
                                    }
                                }
                            }
                            Functional::col2im_ptr(grad_input.get() + inputbatchoffset + inputinoffset , inputtlide.get()
                            , single_input_shape , single_kernel_shape , single_input_stride , single_output_stride , input.device());
                        }
                    }
                }
                return {grad_input};
            }
            std::vector<Tensor<float>> parameters() override{
                return {kernel_};
            }
    };

    template <>
    class Conv<double> : public Module<double>{
        private:
            Tensor<double> kernel_;
            Tensor<double> input_cache;
        public:
            Conv(const Tensor<double> & kernel)
            : kernel_(kernel) {}
            Tensor<double> forward(const std::vector<Tensor<double>> & inputs) override{
                auto input = inputs[0];
                // input :(N , C_in , ...) kernel (C_out , C_in , ...) result (N , C_out , ...)
                if(input.ndim() != kernel_.ndim()){
                    throw std::runtime_error("Conv: input and kernel must have the same number of dimensions");
                }
                if(input.shape()[1] != kernel_.shape()[1]){
                    throw std::runtime_error("Conv: input and kernel must have the same number of input channels");
                }
                if(input.requires_grad()){
                    input_cache = input;
                }
                kernel_.to(input.device());
                auto resultshape = input.shape();
                resultshape[1] = kernel_.shape()[0];
                Tensor<double> result(resultshape , input.device());
                std::vector<size_t> resultstride = result.get_strides();
                auto inputstride = input.get_strides();
                auto kernelstride = kernel_.get_strides();
                std::vector<size_t> single_kernel_shape(kernel_.shape().begin()+2 , kernel_.shape().end());
                std::vector<size_t> single_input_shape(input.shape().begin()+2 , input.shape().end());
                std::vector<size_t> single_input_stride(inputstride.begin() , inputstride.end() - 2);
                std::vector<size_t> single_output_stride(resultstride.begin() , resultstride.end() - 2);
                for(size_t inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                    ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                        for(size_t resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                            ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    Tensor<double> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                    Functional::im2col_ptr<double>(inputtlide.get() , input.get() + inputbatchoffset + inputinoffset
                                     ,single_input_shape , single_kernel_shape ,  single_input_stride , single_output_stride , input.device());
                                    if(result.device() == Cuda){
                                        cublasHandle_t handle;
                                        cublasCreate(&handle);
                                        double alpha = 1.0f;
                                        double beta = 1.0f;
                                        cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                            1 , 
                                            inputstride[input.ndim() - 2],
                                            kernelstride[kernelstride.size() - 2], 
                                            &alpha , 
                                            kernel_.get() + kerneloutoffset + kernelinoffset , 
                                            1, 
                                            inputtlide.get(), 
                                            inputtlide.shape().back()
                                            , 
                                            &beta , 
                                            result.get() + resultbatchoffset + resultoutoffset , 
                                            1
                                        );
                                        cublasDestroy(handle);
                                    }
                                    else{
                                        for(size_t i = 0;i< inputtlide.shape()[0];i++){
                                            double sum = 0;
                                            for(size_t k = 0;k<inputtlide.shape()[1];k++){
                                                sum += inputtlide.get()[i + inputtlide.shape()[1] * k] * 
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
                        std::make_shared<Functional::ModuleFunctionWrapper<double> >(this , input  ));
                }
                return result;
                    
            }
            std::vector<Tensor<double>> _internal_backward(const Tensor<double> & grad_out) override{
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
                Tensor<double>  grad_kernel(kernel_.shape() , kernel_.device());
                for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                    ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                    for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                        for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                            ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                Tensor<double> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                Functional::im2col_ptr<double>(inputtlide.get() , input.get() + inputbatchoffset + inputinoffset
                                    ,single_input_shape , single_kernel_shape ,  single_input_stride , single_output_stride , input.device());
                                if(kernel_.device() == Cuda){
                                    cublasHandle_t handle;
                                    cublasCreate(&handle);
                                    double alpha = 1.0f;
                                    double beta = 1.0f;
                                    cublasDgemm(handle , CUBLAS_OP_N , CUBLAS_OP_T
                                        , 1 , inputtlide.shape()[1] , inputtlide.shape()[0]
                                        , &alpha , grad_out.get() + gradbatchoffset + gradoutoffset, 1 , 
                                        inputtlide.get() , inputtlide.shape().back() , &beta , grad_kernel.get() + kerneloutoffset + kernelinoffset , 1);
                                    cublasDestroy(handle);
                                }
                                else{
                                    for(size_t i = 0;i< inputtlide.shape()[1];i++){
                                        double sum = 0;
                                        for(size_t k = 0;k<inputtlide.shape()[0];k++){
                                            sum += inputtlide.get()[i + inputtlide.shape()[1] * k] * 
                                            grad_out.get()[gradbatchoffset + gradoutoffset + k];
                                        }
                                        grad_kernel.get()[kerneloutoffset + kernelinoffset + i] += sum;
                                    }
                                }
                        }
                    }
                }
                kernel_.set_grad(grad_kernel);
                Tensor<double>  grad_input(input.shape() , input.device());
                for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                    ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                    for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                        for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                        ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                            Tensor<double> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                            if(kernel_.device() == Cuda){
                                cublasHandle_t handle;
                                cublasCreate(&handle);
                                double alpha = 1.0f;
                                double beta = 1.0f;
                                cublasDgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                    ,inputtlide.shape()[1] , inputtlide.shape()[0] , 1
                                    , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                    grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta ,
                                    inputtlide.get() , inputtlide.shape().back()
                                );
                                cublasDestroy(handle);
                            }
                            else{
                                for(size_t i = 0;i<inputtlide.shape()[0];i++){
                                    for(size_t j = 0;j < inputtlide.shape()[1];j++){
                                        inputtlide.get()[i * inputtlide.shape()[1] + j] += 
                                        grad_out.get()[gradbatchoffset + gradoutoffset + i] * 
                                        kernel_.get()[kerneloutoffset + kernelinoffset + j];
                                    }
                                }
                            }
                            Functional::col2im_ptr(grad_input.get() + inputbatchoffset + inputinoffset , inputtlide.get()
                            , single_input_shape , single_kernel_shape , single_input_stride , single_output_stride , input.device());
                        }
                    }
                }
                return {grad_input};
            }
            std::vector<Tensor<double>> parameters() override{
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

    template <typename T>
    __global__ void _softmax_kernel(T * output , const T * input ,const size_t input_size ,const size_t stride){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= input_size){
            return;
        }
        output[index] = exp(input[index]);
        __shared__ T sum[1];
        if(threadIdx.x == 0) sum[0] = 0;
        __syncthreads();
        atomicAdd(sum , output[index]);
        __syncthreads();
        output[index] /= sum[0];

    }

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
                if(output.device() == Cuda){
                    _softmax_kernel<<<input.size() / stride , stride >>>(output.get() , input.get() , input.size()  , stride);
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
    class CrossEntropy : public Module<T>{
        private:
            Softmax<T> softmax;
            Tensor<T> input_cache , input_softmax_cache;
            std::vector<size_t> label_cache_;
        public:
            CrossEntropy(const std::vector<size_t> & label_cache) : label_cache_(label_cache){}
            Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                if(inputs.size()!= 1){
                    throw std::runtime_error("CrossEntropy input size must be 1");
                }
                auto input = inputs[0];
                auto input_softmax = softmax(input);
                input_softmax.to(Cpu);
                if(input.requires_grad()){
                    input_cache = input;
                    input_softmax_cache = input_softmax;
                }
                Tensor<T> loss(T(0));
                size_t batchsize = input.size() / input.shape().back();
                size_t step = input.shape().back();
                for(size_t i = 0;i < batchsize;i++){
                    loss.get()[0] += -log(input_softmax.get()[i * step + label_cache_[i]]);
                }
                loss.get()[0] /= batchsize;
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

}
}


#endif