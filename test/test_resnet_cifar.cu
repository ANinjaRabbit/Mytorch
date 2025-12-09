#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;



int main(){
    DefaultDevice = Cuda;
    auto X = rand<float>({32 , 3 , 64 , 64});
    X.set_requires_grad(true);
    auto y  = rand<float>({32}) * 200;
    auto model = nn::ResNet34<float>(200 , 64 , 64);
    auto params = model.parameters();


    auto optimizer = std::make_shared<optim::AdamW<float>>(params, 0.001);
    auto criterion = nn::CrossEntropy<float>();
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i = 0 ; i < 1000 ; i++){
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
    for(auto & param : params){
        param.get_grad_tensor().print();
    }
}
