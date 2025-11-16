#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    DefaultDevice = Cuda;
    auto x = randn<float>({2 , 3});
    x.print();
    auto y = arange<float>(0 , 10.1 , 2 );
    y.print();
    if(x.device() == Cuda){
        std::cout << "Cuda" << std::endl;
    }
    else{
        std::cout << "Cpu" << std::endl;
    }
}
