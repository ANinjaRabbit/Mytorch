#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    Device device = Cuda;
    DefaultDevice = device;
    nn::Conv2d<float> conv(1 , 2, 3,nn::NoPadding);
    auto kernel = conv.parameters()[0];
    auto bias = conv.parameters()[1];
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
    loss =loss.reshape({label.size()});
    auto l = loss.sum(0);
    l.zero_grad();
    l.backward();
    auto tensor_check = x;
    bias.get_grad_tensor().print();
    l.to(Cpu);
    tensor_check.to(Cpu);
    Tensor<float> grad_ans(tensor_check.shape() , Cpu);
    for(int i = 0;i<2;i++){
        float epsilon = 0.001;
        bias[i] += epsilon;
        bias.to(Cuda);
        auto loss = (conv(x) - label);
        loss = loss * loss;
        loss = loss.reshape({label.size()});
        auto lep = loss.sum(0);
        lep.to(Cpu);
        float grad = (lep.get()[0] - l.get()[0]) / epsilon;

        bias.to(Cpu);
        bias[i] -= epsilon;
        grad_ans[i] = grad;
    }
    grad_ans.print();
}
