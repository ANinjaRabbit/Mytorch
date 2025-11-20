#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    DefaultDevice = Cuda;
    auto x = randn<float>({3 , 3 , 3});
    x.set_requires_grad(true);
    x.print();
    auto y = x.transpose({2 , 0 , 1});
    y.print();
    y.zero_grad();
    auto last_grad = randn<float>({3 , 3 , 3});
    last_grad.print();
    y.backward(last_grad);
    x.get_grad_tensor().print();
}
