#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;




int main(){
    DefaultDevice = Cuda;
    nn::Conv2d<float> conv2d(1 , 3 , 3 , 3);
    conv2d.kernel = ones<float>({3 , 1 , 3 , 3});
    conv2d.bias = zeros<float>({3});

    auto x = ones<float>({1 , 1 , 4 , 4});
    x.set_requires_grad(true);
    x.print();
    auto out = conv2d(x);
    conv2d.zero_grad();
    out.print();
    out.backward();
    x.get_grad_tensor().print();
    conv2d.kernel.get_grad_tensor().print();
     conv2d.bias.get_grad_tensor().print();

}
