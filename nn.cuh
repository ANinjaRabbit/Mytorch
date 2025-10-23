#ifndef MYTORCH_NN_H_
#define MYTORCH_NN_H_

#include "tensor.cuh"
#include <cublas_v2.h>

namespace mytorch{
namespace nn{
    constexpr size_t kCudaTransposeTileSize = 4;
    constexpr size_t kCudaTransposeMaxDim = 16;
    constexpr size_t kCudaIm2colMaxDim = 16;
    class CudaMultiDimIndex{
        private:
            size_t ndim_;
            size_t index_[kCudaIm2colMaxDim];
            size_t shape_[kCudaIm2colMaxDim];
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
        };

        
        template <typename T>
        class NegFunc : public Function<T>{
            public:
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if (inputs.size() != 1){
                        throw std::runtime_error("NegFunc error!");
                    }
                    Tensor<T> result(inputs[0].shape_ , inputs[0].device());
                    if (result.device() == Cuda){
                        _neg_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.data_.get() , inputs[0].data_.get() , result.size() );
                    }
                    else{
                        for(size_t i = 0;i < result.size();i++){
                            result.data_[i] = - inputs[0].data_[i];
                        }
                    }
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T>& grad_output){
                    Tensor<T> gradin(grad_output.shape() , grad_output.device());
                    if(gradin.device() == Cuda){
                        _neg_forward_kernel<<<CudaGetBlocks(gradin.size()) , kCudaThreadsNum>>>(gradin.data_.get() , grad_output.data_.get() , gradin.size());
                    }
                    else{
                        for(size_t i = 0; i < gradin.size();i++){
                            gradin.data_[i] = -grad_output.data_[i];
                        }
                    }
                    return {gradin};
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
            public:
                AddFunc() = default;
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if (inputs.size() != 2 || inputs[0].shape_ != inputs[1].shape_){
                        throw std::runtime_error("AddFunc error!");
                    }
                    if (inputs[0].device() != inputs[1].device()){
                        throw std::runtime_error("AddFunc error!");
                    }
                    Tensor<T> result(inputs[0].shape_ , inputs[0].device());
                    if (result.device() == Cuda){
                        _add_forward_kernel<<<CudaGetBlocks(result.size()), kCudaThreadsNum>>>(result.data_.get(), inputs[0].data_.get(), inputs[1].data_.get() , result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.data_.get()[i] = inputs[0].data_.get()[i] + inputs[1].data_.get()[i];
                        }
                    }
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    return {grad_out.deepcopy() , grad_out.deepcopy()};
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
            public:
                SubFunc() = default;
                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    if (inputs.size() != 2 || inputs[0].shape_ != inputs[1].shape_){
                        throw std::runtime_error("SubFunc error!");
                    }
                    if (inputs[0].device() != inputs[1].device()){
                        throw std::runtime_error("SubFunc error!");
                    }
                    Tensor<T> result(inputs[0].shape_ , inputs[0].device());
                    if (result.device() == Cuda){
                        _sub_forward_kernel<<<CudaGetBlocks(result.size()), kCudaThreadsNum>>>(result.data_.get(), inputs[0].data_.get(), inputs[1].data_.get(),result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.data_.get()[i] = inputs[0].data_.get()[i] - inputs[1].data_.get()[i];
                        }
                    }
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_output) override{
                    return {grad_output.deepcopy() , (-grad_output).deepcopy()};
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
                MulFunc(){
                    a = Tensor<T>();
                    b = Tensor<T>();
                }

                Tensor<T> forward(const std::vector<Tensor<T>> & inputs) override{
                    a = inputs[0].deepcopy() , b = inputs[1].deepcopy();
                    if(inputs.size() != 2 || inputs[0].shape() != inputs[1].shape() || inputs[0].device() != inputs[1].device()){
                        throw std::runtime_error("MulFunc error!");
                    }
                    Tensor<T> result(inputs[0].shape() , inputs[0].device());
                    if(result.device() == Cuda){
                        _mul_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.data_.get() , inputs[0].data_.get() , inputs[1].data_.get(), result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.data_.get()[i] = inputs[0].data_.get()[i] * inputs[1].data_.get()[i];
                        }
                    }
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    Tensor<T> grad_a(grad_out.shape() , grad_out.device());
                    Tensor<T> grad_b(grad_out.shape() , grad_out.device());
                    if (grad_out.device() == Cuda){
                        _mul_forward_kernel<<<CudaGetBlocks(grad_a.size()) , kCudaThreadsNum>>>(grad_a.data_.get() , grad_out.data_.get() , b.data_.get() , grad_a.size());
                        _mul_forward_kernel<<<CudaGetBlocks(grad_b.size()) , kCudaThreadsNum>>>(grad_b.data_.get() , grad_out.data_.get() , a.data_.get() , grad_b.size());
                    }
                    else{
                        for (int i = 0;i<grad_a.size();i++){
                            grad_a.data_.get()[i] = grad_out.data_.get()[i] * b.data_.get()[i];
                            grad_b.data_.get()[i] = grad_out.data_.get()[i] * a.data_.get()[i];
                        }
                    }
                    return {grad_a , grad_b};
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
                    a = inputs[0] , b = inputs[1];
                    if(inputs.size() != 2 || inputs[0].shape() != inputs[1].shape() || inputs[0].device() != inputs[1].device()){
                        throw std::runtime_error("DivFunc error!");
                    }
                    Tensor<T> result(inputs[0].shape() , inputs[0].device());
                    if(result.device() == Cuda){
                        _div_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.data_.get() , inputs[0].data_.get() , inputs[1].data_.get(), result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.data_.get()[i] = inputs[0].data_.get()[i] / inputs[1].data_.get()[i];
                        }
                    }
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    Tensor<T> grad_a(grad_out.shape() , grad_out.device());
                    Tensor<T> grad_b(grad_out.shape() , grad_out.device());
                    if (grad_out.device() == Cuda){
                        _div_backward_kernel_1<<<CudaGetBlocks(grad_a.size()) , kCudaThreadsNum>>>(grad_a.data_.get() , grad_out.data_.get() , b.data_.get() , grad_a.size());
                        _div_backward_kernel_2<<<CudaGetBlocks(grad_b.size()) , kCudaThreadsNum>>>(grad_b.data_.get() , grad_out.data_.get() , a.data_.get() , b.data_.get() , grad_b.size());
                    }
                    else{
                        for (int i = 0;i<grad_a.size();i++){
                            grad_a.data_.get()[i] = grad_out.data_.get()[i] / b.data_.get()[i];
                            grad_b.data_.get()[i] = - grad_out.data_.get()[i] * a.data_.get()[i] / (b.data_.get()[i] * b.data_.get()[i]);
                        }
                    }
                    return {grad_a , grad_b};
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
            public:
                ReLUFunc() = default;
                Tensor<T> forward(const std::vector<Tensor<T>> & input) override{
                    if(input.size() != 1){
                        throw std::runtime_error("ReLUFunc error!");
                    }
                    Tensor<T> result(input[0].shape() , input[0].device());
                    mask = cuda_shared_pointer<bool>(input[0].size() , input[0].device());
                    if( input[0].device() == Cuda){
                        _relu_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.data_.get() , input[0].data_.get() , result.size());
                        _relu_forward_kernel_mask<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(mask.get() , input[0].data_.get() , result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.data_.get()[i] = input[0].data_.get()[i] > 0 ? input[0].data_.get()[i] : 0;
                            mask.get()[i] = input[0].data_.get()[i] > 0;
                        }
                    }
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    Tensor<T> grad_input(grad_out.shape() , grad_out.device());
                    if(grad_out.device() == Cuda){
                        _relu_backward_kernel<<<CudaGetBlocks(grad_input.size()) , kCudaThreadsNum>>>(grad_input.data_.get() , grad_out.data_.get() , mask.get() , grad_input.size());
                    }
                    else{
                        for (int i = 0; i < grad_input.size(); i++){
                            grad_input.data_.get()[i] = mask.get()[i] ? grad_out.data_.get()[i] : 0;
                        }
                    }
                    return {grad_input};
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
            public:
                SigmoidFunc() = default;
                Tensor<T> forward(const std::vector<Tensor<T>> & input) override{
                    if(input.size() != 1){
                        throw std::runtime_error("SigmoidFunc error!");
                    }
                    Tensor<T> result(input[0].shape() , input[0].device());
                    if( input[0].device() == Cuda){
                        _sigmoid_forward_kernel<<<CudaGetBlocks(result.size()) , kCudaThreadsNum>>>(result.data_.get() , input[0].data_.get() , result.size());
                    }
                    else{
                        for (int i = 0; i < result.size(); i++){
                            result.data_.get()[i] = 1 / (1 + std::exp(-input[0].data_.get()[i]));
                        }
                    }
                    output = result.deepcopy();
                    return result;

                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) override{
                    return {(grad_out * (output * (mytorch::ones<T>(output.shape() , output.device()) - output))).deepcopy()};
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
                size_t idx[ kCudaTransposeMaxDim] , tileidx[ kCudaTransposeMaxDim];
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
            public:
                TransposeFunc(const std::vector<size_t> & perm) : perm(perm){}
                Tensor<T> forward(const std::vector<Tensor<T>> & input ) override{
                    if(input.size() != 1)
                        throw std::runtime_error("TransposeFunc error!");
                    std::vector<size_t> newshape = _get_transpose_vec(input[0].shape() , perm);
                    Tensor<T> result(newshape , input[0].device());
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
                            result.data_.get()[outindex] = input[0].data_.get()[index];
                        }
                    }

                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out){
                    return {grad_out.transpose(_get_reverse_perm(perm))};
                }
        };

        template <typename T>
        class ReshapeFunc : public Function<T>{
            private:
                std::vector<size_t> newshape;
                std::vector<size_t> oldshape;
            public:
                ReshapeFunc(const std::vector<size_t> & newshape) : newshape(newshape){}
                Tensor<T> forward(const std::vector<Tensor<T>> & input ) override{
                    oldshape = input[0].shape();
                    if(input.size() != 1 )
                        throw std::runtime_error("ReshapeFunc error!");
                    Tensor<T> result(newshape , input[0].device());
                    result.data_ = input[0].data_;
                    return result;
                }
                std::vector<Tensor<T>> backward(const Tensor<T> & grad_out){
                    return {grad_out.reshape(oldshape)};
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
                    
                    a = input[0];
                    b = input[1];
                    std::vector<size_t> newshape = input[0].shape();
                    newshape[newshape.size() - 1] = input[1].shape()[input[1].shape().size() - 1];
                    Tensor<float> result(newshape , input[0].device());
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
                            cublasSgemm(handle , CUBLAS_OP_T , CUBLAS_OP_T,
                                resultshape[resultshape.size() - 1] , 
                                resultshape[resultshape.size() - 2],
                                input0shape[input0shape.size() - 1] , 
                                &alpha , 
                                input[1].get() + offset1 , 
                                input1stride[2] / input1stride[1] , 
                                input[0].get() + offset0 , 
                                input0stride[2] / input0stride[1] , 
                                &beta , 
                                result.get() + offsetresult , 
                                resultstride[2] / resultstride[1]
                            );
                        }
                        cublasDestroy(handle);
                    }
                    else{
                        for(int offset0 = 0 , offset1 = 0 , offsetresult = 0;offsetresult < result.size();offset0 += step0 , offset1 += step1 , offsetresult += stepresult){
                            for(int i = 0;i<result.shape()[result.shape().size() - 1];i++){
                                for(int j = 0;j<result.shape()[result.shape().size() - 2];j++){
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
                    
                    a = input[0];
                    b = input[1];
                    
                    std::vector<size_t> newshape = input[0].shape();
                    newshape[newshape.size() - 1] = input[1].shape()[input[1].shape().size() - 1];
                    Tensor<double> result(newshape , input[0].device());
                    size_t step0 = input[0].get_strides()[2];
                    size_t step1 = input[1].get_strides()[2];
                    size_t stepresult = result.get_strides()[2];
                    auto resultshape = result.shape();
                    auto input0shape = input[0].shape();
                    auto input1shape = input[1].shape();
                    auto input0stride = input[0].get_strides();
                    auto input1stride = input[1].get_strides();
                    auto resultstride = result.get_strides();
                    if(result.device() == Cuda){
                        cublasHandle_t handle;
                        cublasCreate(&handle);
                        double alpha = 1.0f;
                        double beta = 0.0f;
                        for(int offset0 = 0 , offset1 = 0 , offsetresult = 0;offsetresult < result.size();offset0 += step0 , offset1 += step1 , offsetresult += stepresult){
                            cublasDgemm(handle , CUBLAS_OP_T , CUBLAS_OP_T,
                                resultshape[resultshape.size() - 1] , 
                                resultshape[resultshape.size() - 2],
                                input0shape[input0shape.size() - 1] , 
                                &alpha , 
                                input[1].get() + offset1 , 
                                input1stride[2] / input1stride[1] , 
                                input[0].get() + offset0 , 
                                input0stride[2] / input0stride[1] , 
                                &beta , 
                                result.get() + offsetresult , 
                                resultstride[2] / resultstride[1]
                            );
                        }
                        cublasDestroy(handle);
                    }
                    else{
                        for(int offset0 = 0 , offset1 = 0 , offsetresult = 0;offsetresult < result.size();offset0 += step0 , offset1 += step1 , offsetresult += stepresult){
                            for(int i = 0;i<result.shape()[result.shape().size() - 1];i++){
                                for(int j = 0;j<result.shape()[result.shape().size() - 2];j++){
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
                
        };
        template <typename T>
        class EqFunc : public Function<T>{
            public:
            Tensor<T> forward(const std::vector<Tensor<T>> & input) const{
                return input[0];
            }
            std::vector<Tensor<T>> backward(const Tensor<T> & grad_out) const{
                return {grad_out};
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
                size_t imidx[kCudaIm2colMaxDim];
                for(int i = 0 , index_ = index;i<ndim;i++){
                    imidx[i] = index_ % imshape[ndim - i - 1];
                    index_ /= imshape[ndim - i - 1];
                }
                size_t grid_min[kCudaIm2colMaxDim];
                for(int i = 0;i<ndim;i++){
                    grid_min[i] = imidx[ndim - i - 1] - kernel_shape[i] / 2;
                }
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    bool is_valid = true;
                    size_t kernel_index[kCudaIm2colMaxDim];
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                        if( kernel_index[i] >= kernel_shape[i]){
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
        Tensor<T> im2col(const Tensor<T> & input , const std::vector<size_t> kernel_shape){
            if(kernel_shape.size()!= input.shape().size())
                throw std::runtime_error("im2col: kernel_shape size must be input shape size ");
            if(input.device() == Device::Cpu){
                Tensor<T> result({input.size() , prod_vec(kernel_shape)});
                auto half_kernel_shape = kernel_shape;
                auto instride = input.get_strides();
                auto revinstride = instride;
                auto kernel_size = prod_vec(kernel_shape);
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
                MultiDimIndex index(input.shape());
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
                            if( kernel_index[i] >= input.shape()[i]){
                                is_valid = false;
                                break;
                            }
                        }
                        if(is_valid){
                            size_t input_index = dot_vec(kernel_index , revinstride);
                            size_t result_index = index.calculate_offset(instride) * kernel_size + grid_index.calculate_offset(kernel_stride);
                            result.get()[result_index] = input.get()[input_index];
                        }
                        grid_index.next();
                    }while(!grid_index.is_zero());
                    index.next();
                }while(!index.is_zero());
                return result;
            }
            else{
                Tensor<T> result({input.size() , prod_vec(kernel_shape)} ,Cuda);
                cuda_shared_pointer<size_t> kershape(kernel_shape , Cuda);
                cuda_shared_pointer<size_t> imshape(input.shape() ,Cuda);
                im2col_gpu<<<CudaGetBlocks(input.size()) , kCudaThreadsNum>>>(result.get() , input.get() , result.shape()[1] , input.shape().size()
                ,kershape.get() , imshape.get() ,input.size());
                return result;
            }
        }





    }





}
}


#endif