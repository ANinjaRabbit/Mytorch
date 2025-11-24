#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;



int main(){
    DefaultDevice = Cuda;
    auto X = rand<float>({16 , 3 , 32 , 32});
    X.set_requires_grad(true);
    auto y  = rand<float>({16}) * 10;
    auto model = nn::Sequential<float>({
        std::make_shared<nn::Conv2d<float>>(3 , 6 ,2 , nn::NoPadding) ,
        std::make_shared<nn::BatchNorm2d<float>>(6) ,
        std::make_shared<nn::ReLU<float>>() ,
        std::make_shared<nn::MaxPool2d<float>>(std::vector<size_t>{2 , 2} ) ,
        std::make_shared<nn::Conv2d<float>>(6 , 16 , 5 , nn::NoPadding , Cuda) ,
        std::make_shared<nn::BatchNorm2d<float>>(16) ,
        std::make_shared<nn::ReLU<float>>() ,
        std::make_shared<nn::MaxPool2d<float>>(std::vector<size_t>{2 , 2}) ,
        std::make_shared<nn::Flatten<float>>(1) ,
        std::make_shared<nn::Linear<float>>(5 * 5 * 16, 120) ,
        std::make_shared<nn::ReLU<float>>() ,
        std::make_shared<nn::Linear<float>>(120 , 84) ,
        std::make_shared<nn::ReLU<float>>() ,
        std::make_shared<nn::Linear<float>>(84 , 10)
    });
    auto params = model.parameters();

    auto optimizer = std::make_shared<optim::Adam<float>>(params, 0.02);
    auto scheduler = std::make_shared<optim::lr_scheduler::CosineAnnealingLR<float>>(optimizer , 100 , 0.001);
    auto criterion = nn::CrossEntropy<float>();
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0 ; i < 100 ; i++){
        std::cout << "Iteration " << i << " ";
        auto y_pred = model(X);
        auto loss = criterion({y_pred , y});
        loss.zero_grad();
        loss.backward();
        optimizer->step();
        scheduler->step();
        std::cout << "loss " <<std::fixed<< loss.item() << "\n";

    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);
}
