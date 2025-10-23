#include "nn.cuh"
#include "tensor.cuh"

namespace mytorch{
    template class Tensor<float>;
    template class Tensor<double>;
    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T> & b) const{
        return nn::Functional::AddFunc<T>().forward({*this , b});
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T> & b) const{
        return nn::Functional::SubFunc<T>().forward({*this , b});
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator-() const{
        return nn::Functional::NegFunc<T>().forward({*this} );
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> & b) const{
        return nn::Functional::MulFunc<T>().forward({*this , b});
    }
    template <typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T> & b) const{
        return nn::Functional::DivFunc<T>().forward({*this , b});
    }
    template <typename T>
    Tensor<T> Tensor<T>::relu() const{
        return nn::Functional::ReLUFunc<T>().forward({*this});
    }
    template <typename T>
    Tensor<T> Tensor<T>::sigmoid() const{
        return nn::Functional::SigmoidFunc<T>().forward({*this});
    }
    template <typename T>
    Tensor<T> Tensor<T>::transpose(const std::vector<size_t> & perm) const{
        return nn::Functional::TransposeFunc<T>(perm).forward({*this});
    }
    template <typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<size_t> & newshape) const{
        return nn::Functional::ReshapeFunc<T>(newshape).forward({*this});
    }
    template <typename T>
    Tensor<T> Tensor<T>::matmul(const Tensor<T> & b) const{
        return nn::Functional::MatmulFunc<T>().forward({*this , b});
    }
}