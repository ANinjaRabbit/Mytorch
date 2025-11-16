#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    Device device = Cuda;
    nn::Linear<float> fc1(rand<float>({3 , 3}) , rand<float>({3}));
    nn::ReLU<float> relu;
    nn::Linear<float> fc2(rand<float>({1 , 3}) , rand<float>({1}));
    auto x = rand<float>({ 3} , device);
    x.print();
    x.set_requires_grad(true);
    auto label = rand<float>({1} , device);
    auto y1 = relu(fc1(x));
    y1.print();
    auto y2 = fc2(y1);
    y2.print();
    auto loss = (y2 - label);
    loss.print();
    auto l = loss * loss;
    l.print();
    l.backward();
    x.get_grad_tensor().print();
    x.to(Cpu);
    label.to(Cpu);
    l.to(Cpu);
    for(int i = 0;i<3;i++){
        std::cout << "now " << i << " "  << std::endl;
        float epsilon = 0.001;
        x[i] += epsilon;
        auto y1 = relu(fc1(x));
        auto y2 = fc2(y1);
        auto loss = (y2 - label);
        auto lep = loss * loss;
        float grad = (lep.get()[0] - l.get()[0]) / epsilon;
        x[i] -= epsilon;
        std::cout << "grad check " << i << " " << grad  << std::endl;
    }
}
