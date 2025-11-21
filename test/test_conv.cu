#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
using namespace mytorch;




int main(){
    DefaultDevice = Cuda;

    // correctness test
    auto conv1 = nn::Conv<float>(ones<float>({3 , 3 , 3 , 3}) , nn::NoPadding);
    auto a = ones<float>({3 , 3 , 3 , 3});
    auto b = conv1(a);
    b.print();



    auto conv = nn::Conv<float>({3 , 3 , 3 , 3});
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time = 0;
    for(int i = 0;i<100;i++){
        auto a = randn<float>({3 , 3 , 400 , 400});
        a.set_requires_grad(true);
        cudaEventRecord(start);
        auto b = conv(a);
        b.zero_grad();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    printf("Average Time: %f ms\n", total_time / 100);



}
