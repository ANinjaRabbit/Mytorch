#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;




int main(){
    DefaultDevice = Cuda;
    nn::Conv2d<float> conv2d(2 , 2 ,3 , 3);
    conv2d.kernel = ones<float>({2 , 2 , 3 , 3});
    conv2d.bias = zeros<float>({2});

    auto x = ones<float>({2 , 2 ,4 , 4});
    x.set_requires_grad(true);
    auto out = conv2d(x);
    out.print();

    conv2d.zero_grad();
    out.backward();
    conv2d.kernel.get_grad_tensor().print();
    conv2d.bias.get_grad_tensor().print();
    x.get_grad_tensor().print();


}
