#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"

int main(){
    using namespace mytorch;
    //cpu
    printf("Cpu results\n");
    auto weight = rand<float>({3 , 3});
    auto bias = rand<float>({3});
    auto fc = nn::Linear<float>(weight , bias);
    auto x = rand<float>({2 , 3});
    auto grad_out = rand<float>({2 , 3});
    x.set_requires_grad(true);
    auto y = fc(x);
    y.print();
    y.get_grad_fn()->backward(grad_out)[0].print();
    weight.get_grad_tensor().print();
    bias.get_grad_tensor().print();
    //cuda
    printf("Cuda results\n");
    x.to(Cuda);
    y = fc(x);
    y.print();
    grad_out.to(Cuda);
    y.get_grad_fn()->backward(grad_out)[0].print();
    weight.get_grad_tensor().print();
    bias.get_grad_tensor().print();
}