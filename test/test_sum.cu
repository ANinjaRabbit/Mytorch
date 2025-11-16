#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    auto x = rand<float>({2 , 3 });
    x.set_requires_grad(true);
    auto y = x.sum(0);
    x.print();
    y.print();
}
