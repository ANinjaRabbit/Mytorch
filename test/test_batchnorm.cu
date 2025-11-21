#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"

int main(){
    using namespace mytorch;
    DefaultDevice = Cuda;
    Tensor<float> x = randn<float>({3 , 3 , 3 , 3});
    x.set_requires_grad(true);
    x.print();
    auto bn = nn::BatchNorm2d<float>(3 );
    auto out = bn(x);
    out.print();
    out.zero_grad();
    out.backward();
    x.get_grad_tensor().print();
    bn.parameters()[0].get_grad_tensor().print();
    bn.parameters()[1].get_grad_tensor().print();
}
