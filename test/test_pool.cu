#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    DefaultDevice = Cuda;
    auto x = randn<float>({ 2 ,4 , 4});
    x.set_requires_grad(true);
    auto pool = nn::MaxPool2d<float>({ 2 , 2});
    auto y = pool(x);
    y.zero_grad();
    y.backward(arange<float>(0 , 8 , 1).reshape({2 , 2 , 2}));
    x.get_grad_tensor().print();
    x.print();
    y.print();
}
