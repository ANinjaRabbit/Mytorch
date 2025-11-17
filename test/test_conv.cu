#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    DefaultDevice = Cuda;

    // correctness test
    auto kernel = ones<float>({1 , 1 , 3 , 3});
    auto conv = nn::Conv<float>(kernel , nn::NoPadding);
    auto a = arange<float>(0 , 12 , 1).reshape({1 , 1 , 3 , 4});
    a.print();
    a.set_requires_grad(true);
    auto b = conv(a);
    b.print();
    b.backward();
    a.get_grad_tensor().print();



}
