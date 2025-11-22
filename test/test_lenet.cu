#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;



int main(){
    DefaultDevice = Cuda;
    auto X = rand<float>({10 , 1 , 28 , 28});
    X.set_requires_grad(true);
    auto y = rand<float>({10 ,10});
    auto model = nn::Sequential<float>({
        std::make_shared<nn::Conv2d<float>>(1 , 6 ,5 , nn::NoPadding) ,
        std::make_shared<nn::Pool2d<float>>(std::vector<size_t>{2 , 2}) ,
        std::make_shared<nn::Conv2d<float>>(6 , 16 , 5 , nn::NoPadding , Cuda) ,
        std::make_shared<nn::Pool2d<float>>(std::vector<size_t>{2 , 2}) ,
        std::make_shared<nn::Flatten<float>>(1) ,
        std::make_shared<nn::Linear<float>>(4 * 4 * 16 , 120) ,
        std::make_shared<nn::Linear<float>>(120 , 84) ,
        std::make_shared<nn::Linear<float>>(84 , 10)
    });
    auto params = model.parameters();

    auto optimizer = optim::Adam<float>(params, 0.01);
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0 ; i < 1000 ; i++){
        std::cout << "Iteration " << i << " ";
        auto y_pred = model(X);
        auto loss = (y_pred - y);
        loss = loss * loss;
        loss = loss.sum(1);
        loss.zero_grad();
        loss.backward();
        optimizer.step();
        std::cout << "loss " <<std::fixed<< loss.item() << "\n";

    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);
}
