#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;
int main(){
    DefaultDevice = Cuda;

    // correctness test
    auto kernel = ones<float>({2 , 1 , 3 , 3});
    auto conv = nn::Conv<float>(kernel);
    auto a = arange<float>(0 , 9 , 1).reshape({1 , 1 , 3 , 3});
    a.set_requires_grad(true);
    auto b = conv(a);
    b.backward();
    a.get_grad_tensor().print();
    kernel.get_grad_tensor().print();



    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto x = randn<float>({1 , 1 ,100 , 100});
    x.set_requires_grad(true);
    cudaEventRecord(start);
    auto y = conv(x);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Conv forward time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);
    y.backward();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Conv backward time: " << milliseconds << " ms" << std::endl;
}
