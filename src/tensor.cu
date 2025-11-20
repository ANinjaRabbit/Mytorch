#include "nn.cuh"
#include "tensor.cuh"
#include "autograd.cuh"

namespace mytorch{
    template class Tensor<float>;
    template class Tensor<double>;

    Device DefaultDevice = Cpu;


    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
        if(this->shape() != other.shape()){
            std::cerr << "Tensor shape must be the same" << std::endl;
            throw std::runtime_error("Tensor shape must be the same");
        }
        if(this->requires_grad() || other.requires_grad()){
            auto f = std::make_shared<nn::Functional::AddFunc<T>>();
            Tensor<T> result = f->forward({*this , other});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::AddFunc<T>().forward({*this , other});
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
        if(this->shape() != other.shape()){
            std::cerr << "Tensor shape must be the same" << std::endl;
            throw std::runtime_error("Tensor shape must be the same");
        }
        if(this->requires_grad() || other.requires_grad()){
            auto f = std::make_shared<nn::Functional::SubFunc<T>>();
            Tensor<T> result = f->forward({*this , other});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::SubFunc<T>().forward({*this , other});
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator-() const{
        if(this->requires_grad()){
            auto f = std::make_shared<nn::Functional::NegFunc<T>>();
            Tensor<T> result = f->forward({*this});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::NegFunc<T>().forward({*this});
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {

        if(this->shape() != other.shape()){
            std::cerr << "Tensor shape must be the same" << std::endl;
            throw std::runtime_error("Tensor shape must be the same");
        }
        if(this->requires_grad() || other.requires_grad()){
            auto f = std::make_shared<nn::Functional::MulFunc<T>>();
            Tensor<T> result = f->forward({*this , other});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::MulFunc<T>().forward({*this , other});
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {

        if(this->shape() != other.shape()){
            std::cerr << "Tensor shape must be the same" << std::endl;
            throw std::runtime_error("Tensor shape must be the same");
        }
        if(this->requires_grad() || other.requires_grad()){
            auto f = std::make_shared<nn::Functional::DivFunc<T>>();
            Tensor<T> result = f->forward({*this , other});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::DivFunc<T>().forward({*this , other});
    }

    template <typename T>
    Tensor<T> Tensor<T>::relu() const {
        if(this->requires_grad()){
            auto f =  std::make_shared<nn::Functional::ReLUFunc<T>>();
            Tensor<T> result = f->forward({*this});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::ReLUFunc<T>().forward({*this});
    }

    template <typename T>
    Tensor<T> Tensor<T>::sigmoid() const {
        if(this->requires_grad()){
            auto f =  std::make_shared<nn::Functional::SigmoidFunc<T>>();
            Tensor<T> result = f->forward({*this});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::SigmoidFunc<T>().forward({*this});
    }
    template <typename T>
    Tensor<T> Tensor<T>::transpose(const std::vector<size_t> & perm ) const {
        if(perm.empty()){
            // default permute last two dimensions
            std::vector<size_t> default_perm(this->shape().size());
            if(default_perm.size() < 2){
                throw std::runtime_error("Transpose error: tensor ndim < 2");
            }
            for(int i = 0;i<this->shape().size();i++){
                default_perm[i] = i;
            }
            std::swap(default_perm[default_perm.size() - 1] , default_perm[default_perm.size() - 2]);
            if(this->requires_grad()){
                auto f =  std::make_shared<nn::Functional::TransposeFunc<T>>(default_perm);
                Tensor<T> result = f->forward({*this});
                result.set_grad_fn(f);
                return result;
            }
            return nn::Functional::TransposeFunc<T>(default_perm).forward({*this});
        }
        if(this->shape().size() != perm.size()){
            std::cerr << "Transpose error: tensor ndim != perm size" << std::endl;
            throw std::runtime_error("Transpose error: tensor ndim != perm size");
        }
        if(this->requires_grad()){
            auto f =  std::make_shared<nn::Functional::TransposeFunc<T>>(perm);
            Tensor<T> result = f->forward({*this});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::TransposeFunc<T>(perm).forward({*this});
    }
    template <typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<size_t> & newshape) const {
        if(nn::Functional::prod_vec(newshape) != this->size()){
            std::cerr << "Reshape error: newshape size != tensor size" << std::endl;
            throw std::runtime_error("Reshape error: newshape size != tensor size");
        }
        if(this->requires_grad()){
            auto f =  std::make_shared<nn::Functional::ReshapeFunc<T>>(newshape);
            Tensor<T> result = f->forward({*this});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::ReshapeFunc<T>(newshape).forward({*this});
    }
    template <typename T>
    Tensor<T> Tensor<T>::matmul(const Tensor<T> & other) const {
        if(this->shape().size() != other.shape().size() || this->shape()[this->ndim() - 1] != other.shape()[other.ndim() - 2]){
            std::cerr << "Matmul error: shape mismatch" << std::endl;
            throw std::runtime_error("Matmul error");
        }
        if(this->requires_grad() || other.requires_grad()){
            auto f =  std::make_shared<nn::Functional::MatmulFunc<T>>();
            Tensor<T> result = f->forward({*this , other});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::MatmulFunc<T>().forward({*this , other});
    }
    template <typename T>
    Tensor<T> Tensor<T>::pool2d(const std::vector<size_t> & kernel_shape) const {
        if(this->requires_grad()){
            auto f =  std::make_shared<nn::Functional::Pool2dFunc<T>>(kernel_shape);
            Tensor<T> result = f->forward({*this});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::Pool2dFunc<T>(kernel_shape).forward({*this});
    }
    template <typename T>
    void Tensor<T>::backward(const Tensor<T> & grad_out) {
        auto grad = grad_out.deepcopy();
        if(grad_out.is_null()){
            grad = ones<T>(this->shape() , this->device());
        }
        grad.to(this->device());
        autograd::compute_gradients_of_variables(*this , grad);
    }
    template <typename T>
    Tensor<T> Tensor<T>::sum(const size_t axis) const {
        if(this->requires_grad()){
            auto f =  std::make_shared<nn::Functional::SumFunc<T>>(axis);
            Tensor<T> result = f->forward({*this});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::SumFunc<T>(axis).forward({*this});
    }
    namespace nn{
        __device__ float warpReduceMax(float val) {
            for (int offset = 16; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
            }
            return val;
        }

        __device__ float warpReduceSum(float val) {
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            }
            return val;
        }
        __device__ double warpReduceMax_double(double val) {
            for (int offset = 16; offset > 0; offset /= 2) {
                val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
            }
            return val;
        }
        __device__ double warpReduceSum_double(double val) {
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            }
            return val;
        }

        __global__ void _softmax_kernel_small_512f(float * output , const float * input ,const int N, const int C){
            // for smaller than 512 size softmax
            extern __shared__ float shared[];
            int idx = blockIdx.x; // N
            int tid = threadIdx.x; // C
            int warpId = tid / 32;
            int laneId = tid % 32;

            int warpsPerBlock = blockDim.x / 32; // 512 / 32 = 16

            float * maxvals = shared;
            float * sumvals = &shared[warpsPerBlock];

            const float * x = input + idx * C; // the row to process
            float maxval = tid < C ? x[tid] : -FLT_MAX;
            maxval = warpReduceMax(maxval); // get the warp maximum
            if(laneId== 0) {
                maxvals[warpId] = maxval;
            }
            __syncthreads();
            if(tid == 0){
                maxval = maxvals[0];
                for (int i = 1;i<warpsPerBlock;i++){
                    maxval = fmaxf(maxval , maxvals[i]);
                }
                maxvals[0] = maxval;
            } // get the block maximum
            maxval = maxvals[0];
            if(tid < C){
                output[tid + idx * C] = expf(x[tid] - maxval); // compute exp(x - max)
            }
            __syncthreads();
            float sum = tid < C ? output[tid + idx * C] : 0.0f;
            sum = warpReduceSum(sum); // get the warp sum
            if(laneId== 0) {
                sumvals[warpId] = sum;
            }
            __syncthreads();
            if(tid == 0){
                float val = sumvals[0];
                for (int i = 1;i<warpsPerBlock;i++){
                    val += sumvals[i];
                }
                sumvals[0] = val;
            } // get the block sum
            sum = sumvals[0];
            if(tid < C){
                output[tid + idx * C] /= sum; // normalize
            }
        }

        __global__ void _softmax_kernel_small_512d(double * output , const double * input ,const int N, const int C){
            // for smaller than 512 size softmax
            extern __shared__ double sharedd[];
            int idx = blockIdx.x; // N
            int tid = threadIdx.x; // C
            int warpId = tid / 32;
            int laneId = tid % 32;

            int warpsPerBlock = blockDim.x / 32; // 512 / 32 = 16

            double * maxvals = sharedd;
            double * sumvals = &sharedd[warpsPerBlock];

            const double * x = input + idx * C; // the row to process
            double maxval = tid < C ? x[tid] : -DBL_MAX;
            maxval = warpReduceMax_double(maxval); // get the warp maximum
            if(laneId== 0) {
                maxvals[warpId] = maxval;
            }
            __syncthreads();
            if(tid == 0){
                maxval = maxvals[0];
                for (int i = 1;i<warpsPerBlock;i++){
                    maxval = fmax(maxval , maxvals[i]);
                }
                maxvals[0] = maxval;
            } // get the block maximum
            maxval = maxvals[0];
            if(tid < C){
                output[tid + idx * C] = exp(x[tid] - maxval); // compute exp(x - max)
            }
            __syncthreads();
            double sum = tid < C ? output[tid + idx * C] : 0.0;
            sum = warpReduceSum_double(sum); // get the warp sum
            if(laneId== 0) {
                sumvals[warpId] = sum;
            }
            __syncthreads();
            if(tid == 0){
                double val = sumvals[0];
                for (int i = 1;i<warpsPerBlock;i++){
                    val += sumvals[i];
                }
                sumvals[0] = val;
            } // get the block sum
            sum = sumvals[0];
            if(tid < C){
                output[tid + idx * C] /= sum; // normalize
            }
        }

        __global__ void _softmax_kernel_general_f(float * output , const float * input , const int N , const int C){
            extern __shared__ float shared[];
            int idx = blockIdx.x;
            int tid = threadIdx.x;
            int warpId = threadIdx.x / 32; 
            int laneId = threadIdx.x % 32;

            int warpsPerBlock = blockDim.x / 32;

            float* maxvals = shared;
            float* sumvals = &shared[warpsPerBlock];

            const float* x = input + idx * C;

            float maxval = -FLT_MAX;
            for (int i = tid; i < C; i += blockDim.x) {
                maxval = fmaxf(maxval, x[i]); 
            }
            maxval = warpReduceMax(maxval);
            if (laneId == 0) 
                maxvals[warpId] = maxval;
            __syncthreads();
            if (tid == 0) {
                float val = maxvals[tid];
                for (int i = 1; i < warpsPerBlock; i++) {
                    val = fmaxf(val, maxvals[i]);
                }
                // store the final max in the first position
                maxvals[0] = val;
            }
            __syncthreads();
            maxval = maxvals[0];
            float sum = 0.0f;
            for (int i = tid; i < C; i += blockDim.x) {
                output[i + idx * C] = expf(x[i] - maxval);
                sum += output[i + idx * C];
            }   
            __syncthreads();
            sum = warpReduceSum(sum);
            if( laneId == 0 ) 
                sumvals[warpId] = sum;
            __syncthreads();
            if (tid == 0) {
                float val = sumvals[0];
                for (int i = 1; i < warpsPerBlock; i++) {
                    val += sumvals[i];
                }
                sumvals[0] = val;
            }
            sum = sumvals[0];
            for (int i = tid; i < C; i += blockDim.x) {
                output[i + idx * C] /= sum;
            }
            
        }

        __global__ void _softmax_kernel_general_d(double * output , const double * input , const int N , const int C){
            extern __shared__ double sharedd[];
            int idx = blockIdx.x;
            int tid = threadIdx.x;
            int warpId = threadIdx.x / 32; 
            int laneId = threadIdx.x % 32;

            int warpsPerBlock = blockDim.x / 32;

            double* maxvals = sharedd;
            double* sumvals = &sharedd[warpsPerBlock];

            const double* x = input + idx * C;

            double maxval = -FLT_MAX;
            for (int i = tid; i < C; i += blockDim.x) {
                maxval = fmax(maxval, x[i]); 
            }
            maxval = warpReduceMax_double(maxval);
            if (laneId == 0) 
                maxvals[warpId] = maxval;
            __syncthreads();
            if (tid == 0) {
                double val = maxvals[tid];
                for (int i = 1; i < warpsPerBlock; i++) {
                    val = fmax(val, maxvals[i]);
                }
                // store the final max in the first position
                maxvals[0] = val;
            }
            __syncthreads();
            maxval = maxvals[0];
            double sum = 0.0f;
            for (int i = tid; i < C; i += blockDim.x) {
                output[i + idx * C] = exp(x[i] - maxval);
                sum += output[i + idx * C];
            }   
            __syncthreads();
            sum = warpReduceSum_double(sum);
            if( laneId == 0 ) 
                sumvals[warpId] = sum;
            __syncthreads();
            if (tid == 0) {
                double val = sumvals[0];
                for (int i = 1; i < warpsPerBlock; i++) {
                    val += sumvals[i];
                }
                sumvals[0] = val;
            }
            sum = sumvals[0];
            for (int i = tid; i < C; i += blockDim.x) {
                output[i + idx * C] /= sum;
            }
            
        }
        namespace Functional {
            __global__ void _sum_forward_kernel_f(float * output , const float * input ,  const size_t reduce , const size_t inner){
                size_t ridx = threadIdx.x;
                size_t iidx = blockIdx.x % inner;
                size_t oidx = blockIdx.x / inner;
                extern __shared__ float smem_sum_f[];
                size_t warpId = ridx / 32;
                size_t laneId = ridx % 32;
                size_t warpsPerBlock = blockDim.x / 32;
                float sum = 0.0f;
                for(size_t i = ridx;i<reduce;i+=blockDim.x){
                    sum += input[oidx * reduce + i * inner + iidx];
                }
                sum = warpReduceSum(sum);
                if(laneId == 0) smem_sum_f[warpId] = sum;
                __syncthreads();
                if(ridx == 0){
                    float val = smem_sum_f[0];
                    for(size_t i = 1;i<warpsPerBlock;i++){
                        val += smem_sum_f[i];
                    }
                    output[oidx * reduce + iidx] = val;
                }

            }
            __global__ void _sum_forward_kernel_d(double * output , const double * input ,  const size_t reduce , const size_t inner){
                size_t ridx = threadIdx.x;
                size_t iidx = blockIdx.x % inner;
                size_t oidx = blockIdx.x / inner;
                extern __shared__ double smemd[];
                size_t warpId = ridx / 32;
                size_t laneId = ridx % 32;
                size_t warpsPerBlock = blockDim.x / 32;
                double sum = 0.0f;
                for(size_t i = ridx;i<reduce;i+=blockDim.x){
                    sum += input[oidx * reduce + i * inner + iidx];
                }
                sum = warpReduceSum_double(sum);
                if(laneId == 0) smemd[warpId] = sum;
                __syncthreads();
                if(ridx == 0){
                    double val = smemd[0];
                    for(size_t i = 1;i<warpsPerBlock;i++){
                        val += smemd[i];
                    }
                    output[oidx * reduce + iidx] = val;
                }
            }

        }

    }


}