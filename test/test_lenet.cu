#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;



int main(){
    DefaultDevice = Cuda;
    auto X = randn<float>({100 , 3 , 10 , 10});
    X.set_requires_grad(true);
    auto y  = rand<float>({100}) * 10;
    auto model = nn::Sequential<float>({
        std::make_shared<nn::Conv2d<float>>(3 , 2 ,3 , 3) ,
        std::make_shared<nn::BatchNorm2d<float>>(2) ,
        std::make_shared<nn::ReLU<float>>() ,
        std::make_shared<nn::MaxPool2d<float>>(std::vector<size_t>{2 , 2}) ,
        std::make_shared<nn::Flatten<float>>(1) ,
        std::make_shared<nn::Linear<float>>(4 * 4* 2 , 10) ,
        std::make_shared<nn::ReLU<float>>() ,
        std::make_shared<nn::Linear<float>>(10 , 84) ,
        std::make_shared<nn::ReLU<float>>() ,
        std::make_shared<nn::Linear<float>>(84 , 10)
    });
    auto params = model.parameters();


    auto optimizer = std::make_shared<optim::Adam<float>>(params, 0.001);
    auto criterion = nn::CrossEntropy<float>();
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0 ; i < 1000 ; i++){
        std::cout << "Iteration " << i << " ";
        optimizer->zero_grad();
        auto y_pred = model(X);
        auto loss = criterion({y_pred , y});
        loss.backward();
        optimizer->step();
        std::cout << "loss " <<std::fixed<< loss.item() << "\n";

    }
    for(auto & param : params){
        param.get_grad_tensor().print();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);
}
