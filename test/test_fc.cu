#include <iostream>
#include "../src/tensor.cuh"
#include "../src/nn.cuh"

int main(){
    using namespace mytorch;
    DefaultDevice = Cuda;



    auto weight = rand<float>({10 , 100});
    auto bias = rand<float>({10});
    auto fc = nn::Linear<float>(weight , bias);
    auto x = rand<float>({100 , 100});
    x.set_requires_grad(true);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    auto y = fc(x);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Linear forward time: " << milliseconds << " ms" << std::endl;
    y.print();

    cudaEventRecord(start);
    y.backward();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Linear backward time: " << milliseconds << " ms" << std::endl;
}