#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    DefaultDevice = Cuda;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto x = arange<float>(0 , 17 * 17 , 1).reshape({17 , 17});
    x.print();
    x.set_requires_grad(true);
    cudaEventRecord(start);
    auto y = x.transpose({1 , 0});
    y.backward();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    y.print();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Transpose time: " << milliseconds << " ms" << std::endl;
}
