#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    DefaultDevice = Cuda;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto x = rand<float>({400 , 400});
    x.set_requires_grad(true);
    auto pool = nn::Pool2d<float>({ 3 , 3});
    cudaEventRecord(start);
    auto y = pool(x);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Pool2d time: " << milliseconds << " ms" << std::endl;
}
