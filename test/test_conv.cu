#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;




int main(){
    DefaultDevice = Cuda;

    // correctness test
    auto conv1 = nn::Conv2d<float>(3 , 3 , 3 );
    auto a = randn<float>({1 , 3 , 400 , 400});

    float tot_time = 0;
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i = 0;i < 100;i++){
        std::cout << "Iteration " << i << " ";
        cudaEventRecord(start);
        try{
            auto b = conv1(a);
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        tot_time += milliseconds;
        std::cout << "Time: " << milliseconds << " ms\n";
    }
    printf("Time: %f ms\n", tot_time / 100);
}
