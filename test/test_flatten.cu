#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"

int main(){
    using namespace mytorch;
    DefaultDevice = Cuda;

    auto x = ones<float>({3 , 3 , 3});
    x.set_requires_grad(true);
    auto flatten = nn::Flatten<float>();
    auto y = flatten(x);
    y.print();
    y.zero_grad();
    y.backward();
    x.get_grad_tensor().print();
}