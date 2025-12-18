#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;



int main(){
    DefaultDevice = Cuda;
    auto X = ones<float>({32 , 3 , 32 , 32});
    X.set_requires_grad(true);
    auto y  = ones<float>({32}) * 9;
    auto model = nn::ResNet18<float>(10 , 32 , 32);
    model.load("resnet18_cifar10.pth");
    auto params = model.parameters();


    auto optimizer = std::make_shared<optim::AdamW<float>>(params, 0.001);
    auto criterion = nn::CrossEntropy<float>();
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i = 0 ; i < 10 ; i++){
        std::cout << "Iteration " << i << " ";
        optimizer->zero_grad();
        cudaEventRecord(start);
        auto y_pred = model(X);
        auto loss = criterion({y_pred , y});
        loss.backward();
        std::cout << "loss "  << std::endl;
        optimizer->step();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time: %f ms\n", milliseconds);
        std::cout << "loss " <<std::fixed<< loss.item() << "\n";

    }

    model.save("resnet18_cifar10.pth");

}
