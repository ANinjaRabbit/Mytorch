#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    Device device = Cuda;
    DefaultDevice = device;
    nn::BatchNorm2d<float> bn(2);
    auto gamma = bn.parameters()[0];
    auto beta = bn.parameters()[1];
    auto x = rand<float>({ 1 ,2 , 4 ,4} , device);
     x.print();
    x.set_requires_grad(true);
    auto label = rand<float>({1 , 2 , 4 ,4} , device);
    auto y1 = bn(x);
    y1.print();
    auto loss = (y1 - label);
    loss.print();
    loss = loss * loss;
    loss.print();
    loss =loss.reshape({label.size()});
    auto l = loss.sum(0);
    l.zero_grad();
    l.backward();
    auto check_tensor = x;
    check_tensor.get_grad_tensor().print();
    l.to(Cpu);
    check_tensor.to(Cpu);
    Tensor<float> gradans(check_tensor.shape() , Cpu);
    for(int i = 0;i<check_tensor.size();i++){
        float epsilon = 0.001;
        check_tensor[i] += epsilon;
        check_tensor.to(Cuda);
        auto loss = (bn(x) - label);
        loss = loss * loss;
        loss = loss.reshape({label.size()});
        auto lep = loss.sum(0);
        lep.to(Cpu);
        float grad = (lep.get()[0] - l.get()[0]) / epsilon;

        check_tensor.to(Cpu);
        check_tensor[i] -= epsilon;
        gradans[i] = grad;
    }
    gradans.print();
}
