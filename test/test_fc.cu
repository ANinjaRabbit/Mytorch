#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"

int main(){
    using namespace mytorch;

    auto x = ones<float>({3});
    x.set_requires_grad(true);
    auto weight = ones<float>({3 , 3});
    auto bias = ones<float>({3});
    auto fc = nn::Linear<float>(weight , bias);
    auto y = fc(x);
    x.print();
    y.print();
    x.zero_grad();
    y.backward();
    x.get_grad_tensor().print();
    weight.get_grad_tensor().print();
    bias.get_grad_tensor().print();
}