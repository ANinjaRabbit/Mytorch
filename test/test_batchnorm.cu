#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    DefaultDevice = Cpu;
    auto a = rand<float>({ 1 , 2 , 4 , 4});
    a.set_requires_grad(true);
    nn::BatchNorm2d<float> bn(2);
    auto b = bn(a);
    a.print();
    b.print();
    b.zero_grad();
    b.backward();
    a.get_grad_tensor().print();
    bn.gamma.get_grad_tensor().print();
    bn.beta.get_grad_tensor().print();
}
