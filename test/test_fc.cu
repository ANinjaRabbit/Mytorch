#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"

int main(){
    using namespace mytorch;
    DefaultDevice = Cuda;

    auto x = arange<float>(0,9 , 1).reshape({3 , 3});
    x.set_requires_grad(true);
    auto weight = arange<float>(0,6 , 1).reshape({2 , 3});
    auto bias = ones<float>({2});
    auto fc = nn::Linear<float>(weight , bias);
    fc.zero_grad();
    auto y = fc(x);
    x.print();
    y.print();
    y.backward();
    x.get_grad_tensor().print();
    weight.get_grad_tensor().print();
    bias.get_grad_tensor().print();
}