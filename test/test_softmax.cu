#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;

int main(){
    DefaultDevice = Cuda;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto softmax = nn::Softmax<float>();
    auto X = ones<float>({10 , 512});
    cudaEventRecord(start);
    auto y = softmax(X);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    return 0;
}