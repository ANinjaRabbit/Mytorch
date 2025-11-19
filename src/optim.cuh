#ifndef _OPTIM_H_
#define _OPTIM_H_
#include "tensor.cuh"
#include "nn.cuh"

namespace mytorch{
    namespace optim{

        template <typename T>
        __global__ void _sgd_step_kernel(T * param, const T * grad,const T lr , const size_t size){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < size){
                param[idx] -= lr * grad[idx];
            }
        }
        
        template <typename T>
        class SGD{
                T lr_;
                std::vector<Tensor<T>> params_;
                Device device_;
            public:
                SGD(std::vector<Tensor<T>> & params, T lr = T(0.01), Device device = DefaultDevice) : lr_(lr) , params_(params) , device_(device){}
                void step(){
                    if(device_ == Device::Cpu){
                        for(auto & param : params_)
                            for(size_t i = 0 ; i < param.size() ; i++)
                                param.get()[i] -= lr_ * param.get_grad()[i];
                    }
                    else{
                        for(auto & param : params_){

                            _sgd_step_kernel<<<CudaGetBlocks(param.size()) , kCudaThreadsNum>>>(
                                param.get(),
                                param.get_grad().get(),
                                lr_,
                                param.size()
                            );
                        }
                    }

                }

        };

        template <typename T>
        __global__ void _adam_step_kernel(
            T * param,
            T * m,
            T * v,
            const T * grad,
            const size_t t,
            const T beta1,
            const T beta2,
            const T lr,
            const T eps,
            const size_t size
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < size){
                m[idx] = beta1 * m[idx] + (T(1) - beta1) * grad[idx];
                v[idx] = beta2 * v[idx] + (T(1) - beta2) * grad[idx] * grad[idx];
                T m_hat = m[idx] / (T(1) - pow(beta1, t));
                T v_hat = v[idx] / (T(1) - pow(beta2, t));
                param[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
            }
        }
        template <>
        __global__ void _adam_step_kernel<float>(
            float * param,
            float * m,
            float * v,
            const float * grad,
            const size_t t,
            const float beta1,
            const float beta2,
            const float lr,
            const float eps,
            const size_t size
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < size){
                m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad[idx];
                v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad[idx] * grad[idx];
                float m_hat = m[idx] / (1.0 - powf(beta1, t));
                float v_hat = v[idx] / (1.0 - powf(beta2, t));
                param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
            }
        }
        template <typename T>
        class Adam{
                T lr_;
                T beta1_;
                T beta2_;
                T eps_;
                std::vector<Tensor<T>> params_;
                std::vector<Tensor<T>> m_;
                std::vector<Tensor<T>> v_;
                size_t t_;
                Device device_;
            public:
                Adam(std::vector<Tensor<T>> & params,const T lr = T(0.001) ,const T beta1 = T(0.9) , const T beta2 = T(0.999) , const T eps = T(1e-8)  , Device device = DefaultDevice) : lr_(lr) , beta1_(beta1) , beta2_(beta2) , eps_(eps) , params_(params) , device_(device){
                    for(auto & param : params_){
                        m_.emplace_back(zeros<T>(param.shape() , param.device()));
                        v_.emplace_back(zeros<T>(param.shape() , param.device()));
                    }
                    t_ = 0;
                }
                void step(){
                    t_++;
                    if(device_ == Device::Cuda){
                        for(auto i = 0 ; i < params_.size() ; i++){
                            auto & param = params_[i];
                            auto & m = m_[i];
                            auto & v = v_[i];
                            _adam_step_kernel<<<CudaGetBlocks(param.size()) , kCudaThreadsNum>>>(
                                param.get(),
                                m.get(),
                                v.get(),
                                param.get_grad().get(),
                                t_,
                                beta1_,
                                beta2_,
                                lr_,
                                eps_,
                                param.size()
                            );
                        }
                    }
                    else{
                        for(auto i = 0 ; i < params_.size() ; i++){
                            auto & param = params_[i];
                            auto & m = m_[i];
                            auto & v = v_[i];
                            auto grad = param.get_grad_tensor();
                            m = beta1_ * m + (T(1) - beta1_) * grad;
                            v = beta2_ * v + (T(1) - beta2_) * grad * grad;
                            for(size_t j = 0 ; j < param.size() ; j++){
                                param.get()[j] -= lr_ * m.get()[j] / (T(1) - powf(beta1_, t_)) * (sqrtf(v.get()[j] / (T(1) - powf(beta2_, t_))) + eps_);
                            }
                        }

                    }
                }
        };
    }
}

#endif