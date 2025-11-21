#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;




int main(){
    DefaultDevice = Cuda;

    // correctness test
    auto conv1 = nn::Conv<float>(randn<float>({3 , 3 , 5 , 5}) , nn::NoPadding);
    auto a = randn<float>({3 , 3 , 6 , 6});
    a.set_requires_grad(true);

    auto b = conv1(a);
    b.print();



    auto conv = nn::Conv<float>({3 , 3 , 5 , 5} , nn::NoPadding);
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    a = randn<float>({3 , 3 , 6 , 6});
    auto label = randn<float>({3 , 3 , 2 , 2});
    a.set_requires_grad(true);
    float total_time = 0;
    auto optim = optim::Adam<float>(conv.parameters(), 0.001);
    for(int i = 0;i<100;i++){
        cudaEventRecord(start);
        auto b = conv(a);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
        auto loss = (b - label);
        loss = loss * loss;
        loss = loss.reshape({36});
        loss = loss.sum(0);
        std::cout << "loss " << loss.item() << "\n";
        b.zero_grad();
        b.backward();
        optim.step();
    }
    printf("Average Time: %f ms\n", total_time / 100);




}
