#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    Device device = Cuda;
    DefaultDevice = device;
    auto kernel = rand<float>({2 , 2,3 , 3});
    nn::Conv<float> conv(kernel,nn::NoPadding);
    auto x = rand<float>({ 1 ,2 , 4 ,4} , device);
    x.print();
    x.set_requires_grad(true);
    auto label = rand<float>({1 , 2 , 2 ,2} , device);
    auto y1 = conv(x);
    y1.print();
    auto loss = (y1 - label);
    loss.print();
    loss = loss * loss;
    loss.print();
    loss =loss.reshape({4 * 2});
    auto l = loss.sum(0);
    l.zero_grad();
    l.backward();
    kernel.get_grad_tensor().print();
    l.to(Cpu);
    kernel.to(Cpu);
    for(int i = 0;i<3;i++){
        std::cout << "now " << i << " "  << std::endl;
        float epsilon = 0.001;
        kernel[i] += epsilon;
        kernel.to(Cuda);
        auto loss = (conv(x) - label);
        loss = loss * loss;
        loss = loss.reshape({4 * 2});
        auto lep = loss.sum(0);
        lep.to(Cpu);
        float grad = (lep.get()[0] - l.get()[0]) / epsilon;

        kernel.to(Cpu);
        kernel[i] -= epsilon;
        std::cout << "grad check " << i << " " << grad  << std::endl;
    }
}
