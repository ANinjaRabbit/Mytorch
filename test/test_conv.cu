#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;




int main(){
    DefaultDevice = Cpu;
    nn::Conv2d<float> conv2d(2 , 2 , 3 , nn::NoPadding);
    conv2d.kernel = ones<float>({2 , 2 , 3 , 3});
    conv2d.bias = ones<float>({2});
    auto x = ones<float>({2 , 2 , 5 , 5});
    x.set_requires_grad(true);
    auto out = conv2d(x);
    std::cout << "out grad" << std::endl;
    out.backward(arange<float>(0 , 2 * 2 * 3 * 3 , 1).reshape({2 , 2 , 3 , 3}));
    x.get_grad_tensor().print();

}
