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
    }
}

#endif