#ifndef _OPTIM_H_
#define _OPTIM_H_
#include "tensor.cuh"
#include "nn.cuh"
#define PI 3.14159265358979323846


namespace mytorch{
    namespace optim{


        namespace lr_scheduler {
            template <typename T>
            class StepLR;
            template <typename T>
            class CosineAnnealingLR;
        } // namespace lr_scheduler

        // base class for optimizers
        template <typename T>
        class Optimizer{
            protected:
                T lr_;
                T weight_decay_;
            public:
                friend class lr_scheduler::StepLR<T>;
                friend class lr_scheduler::CosineAnnealingLR<T>;
                Optimizer(T lr = T(0.01) , T weight_decay = T(0.0)) : lr_(lr) , weight_decay_(weight_decay){}
                void step(){};
        };

        template <typename T>
        __global__ void _sgd_step_kernel(T * param, const T * grad,const T lr , const T weight_decay , const size_t size){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < size){
                param[idx] -= lr * (grad[idx] +  weight_decay * param[idx]);
            }
        }
        
        template <typename T>
        class SGD: public Optimizer<T>{
                std::vector<Tensor<T>> params_;
                Device device_;
            public:
                SGD(std::vector<Tensor<T>> & params, T lr = T(0.01) , T weight_decay = T(0.0) , Device device = DefaultDevice) : Optimizer<T>(lr , weight_decay) , params_(params) , device_(device){}
                void step(){
                    if(device_ == Device::Cpu){
                        for(auto & param : params_)
                            for(size_t i = 0 ; i < param.size() ; i++)
                                param.get()[i] -= this->lr_ * param.get_grad()[i] + this->lr_ * this->weight_decay_ * param.get()[i];
                    }
                    else{
                        for(auto & param : params_){

                            _sgd_step_kernel<<<CudaGetBlocks(param.size()) , kCudaThreadsNum>>>(
                                param.get(),
                                param.get_grad().get(),
                                this->lr_,
                                this->weight_decay_,
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
            const T weight_decay,
            const size_t size
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < size){
                m[idx] = beta1 * m[idx] + (T(1) - beta1) * grad[idx];
                v[idx] = beta2 * v[idx] + (T(1) - beta2) * grad[idx] * grad[idx];
                T m_hat = m[idx] / (T(1) - pow(beta1, t));
                T v_hat = v[idx] / (T(1) - pow(beta2, t));
                param[idx] -= lr * m_hat / (sqrt(v_hat) + eps) + lr * weight_decay * param[idx];
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
            const float weight_decay,
            const size_t size
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < size){
                m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad[idx];
                v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad[idx] * grad[idx];
                float m_hat = m[idx] / (1.0 - powf(beta1, t));
                float v_hat = v[idx] / (1.0 - powf(beta2, t));
                param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps) + lr * weight_decay * param[idx];
            }
        }
        template <typename T>
        class Adam : public Optimizer<T>{
                T beta1_;
                T beta2_;
                T eps_;
                std::vector<Tensor<T>> params_;
                std::vector<Tensor<T>> m_;
                std::vector<Tensor<T>> v_;
                size_t t_;
                Device device_;
            public:
                Adam(std::vector<Tensor<T>> & params,const T lr = T(0.001) ,const T beta1 = T(0.9) , const T beta2 = T(0.999) , const T eps = T(1e-8) , const T weight_decay = T(0) , Device device = DefaultDevice) : Optimizer<T>(lr , weight_decay) , beta1_(beta1) , beta2_(beta2) , eps_(eps) , params_(params) , device_(device){
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
                                this->lr_,
                                eps_,
                                this->weight_decay_,
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
                                param.get()[j] -= this->lr_ * m.get()[j] / (T(1) - powf(beta1_, t_)) * (sqrtf(v.get()[j] / (T(1) - powf(beta2_, t_))) + eps_) + this->lr_ * this->weight_decay_ * param.get()[j];
                            }
                        }

                    }
                }
        };

        namespace lr_scheduler{
            template <typename T>
            class StepLR{
                std::shared_ptr<Optimizer<T>> optimizer_;
                T gamma_;
                size_t step_size_;
                size_t last_epoch_;
                public:
                    StepLR(std::shared_ptr<Optimizer<T>> optimizer , T gamma = T(0.1) , size_t step_size = 1) : optimizer_(optimizer) , gamma_(gamma) , step_size_(step_size) , last_epoch_(0){}
                    void step(){
                        last_epoch_++;
                        if(last_epoch_ % step_size_ == 0){
                            optimizer_->lr_ *= gamma_;
                        }
                    }
            };
            
            template <typename T>
            class CosineAnnealingLR{
                std::shared_ptr<Optimizer<T>> optimizer_;
                T T_max_;
                T eta_min_;
                size_t last_epoch_;
                public:
                    CosineAnnealingLR(std::shared_ptr<Optimizer<T>> optimizer , T T_max , T eta_min = T(0)) : optimizer_(optimizer) , T_max_(T_max) , eta_min_(eta_min) , last_epoch_(0){}
                    void step(){
                        last_epoch_++;
                        T cos_val = cosf(PI * static_cast<T>(last_epoch_) / T_max_);
                        T lr = eta_min_ + (optimizer_->lr_ - eta_min_) * (T(1) + cos_val) / T(2);
                        optimizer_->lr_ = lr;
                    }
            };
        }
    }
}

#endif