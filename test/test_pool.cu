#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    DefaultDevice = Cuda;
    auto x = rand<float>({128 , 512 ,4 , 4});
    x.set_requires_grad(true);
    auto y = x.maxpool2d(4 , 4);
    y.zero_grad();
    y.backward(ones<float>({128 , 512 , 1 , 1}));
}
