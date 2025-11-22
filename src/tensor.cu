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


}