#include "nn.cuh"
#include "tensor.cuh"

namespace mytorch{
    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
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
    Tensor<T> Tensor<T>::transpose(const std::vector<size_t> & perm) const {
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
        if(this->requires_grad() || other.requires_grad()){
            auto f =  std::make_shared<nn::Functional::MatmulFunc<T>>();
            Tensor<T> result = f->forward({*this , other});
            result.set_grad_fn(f);
            return result;
        }
        return nn::Functional::MatmulFunc<T>().forward({*this , other});
    }
    template class Tensor<float>;
    template class Tensor<double>;

}