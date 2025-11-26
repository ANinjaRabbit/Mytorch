#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;



int main(){
    DefaultDevice = Cpu;
    auto X = arange<float>(0 , 100 , 1).reshape({10 , 10});
    X.set_requires_grad(true);
    auto y = ones<float>({10});
    auto criterion = nn::CrossEntropy<float>();
    auto loss = criterion({X , y});
    std::cout << loss.item() << "\n";
    loss.zero_grad();
    loss.backward();
    X.get_grad_tensor().print();
    DefaultDevice = Cuda;
    X = arange<float>(0 , 100 , 1).reshape({10 , 10});
    X.set_requires_grad(true);
    y = ones<float>({10});
    criterion = nn::CrossEntropy<float>();
    loss = criterion({X , y});
    std::cout << loss.item() << "\n";
    loss.zero_grad();
    loss.backward();
    X.get_grad_tensor().print();
}